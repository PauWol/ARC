from dataclasses import dataclass, field
import difflib
import time
from typing import Callable
import uuid
import logging

from src.llama_runtime import LlamaRuntime

from src.memory import build_initial_state, AgentState
from src.roles import (
    TaskExtractor,
    ExtractedTask,
    Planner,
    Plan,
    Validator,
    build_validator_prompt,
)
from src.roles.synthesizer import Synthesizer
from src.schema import ToolResult
from src.tools.utils import tool_result_validator

from src.llama_runtime import ModelSource, RuntimeOptions
from src.tools.sandbox.policy import SandboxPolicy
from src.tools.builtin import make_builtin_tools

from src.events import (
    EventBus,
    emit_agent_done,
    emit_agent_error,
    emit_agent_started,
    emit_action_finished,
    emit_action_planned,
    emit_action_started,
    emit_context_built,
    emit_intent_extracted,
    emit_state_updated,
    emit_step_started,
    emit_synthesis_finished,
    emit_synthesis_started,
    emit_thinking_started,
    emit_tool_error,
    emit_validation_finished,
    emit_validation_started,
    emit_reasoning_started,
    emit_reasoning_chunk,
    emit_reasoning_finished,
)

# Common near-misses a model reaches for that don't match a registered tool
# name verbatim (e.g. planner prompts mentioning "bash" informally while the
# actual registered tool is "run_bash"). Extend as you add/rename tools.
TOOL_ALIASES: dict[str, str] = {
    "bash": "run_bash",
    "shell": "run_bash",
    "sh": "run_bash",
    "python": "execute_python",
    "py": "execute_python",
    "exec": "execute_python",
    "read": "read_file",
    "cat": "read_file",
    "write": "write_file",
    "save": "write_file",
}

logger = logging.getLogger("agent")

class ReasoningHook:
    """
    Bridges BaseRole's reasoning stream to the Agent's event bus so a UI can
    render "thinking..." live. One instance per role; reads run_id/state off
    the Agent at call time since those change every run() but the role
    instances (and therefore their hooks) are constructed once in __init__.
    """

    def __init__(self, agent: "Agent", role: str) -> None:
        self._agent = agent
        self._role = role

    def start(self) -> None:
        emit_reasoning_started(
            self._agent.event_bus,
            run_id=self._agent._current_run_id,
            state=self._agent.state,
            role=self._role,
        )

    def chunk(self, text: str) -> None:
        emit_reasoning_chunk(
            self._agent.event_bus,
            run_id=self._agent._current_run_id,
            state=self._agent.state,
            role=self._role,
            chunk=text,
        )

    def finish(self, full_text: str) -> None:
        emit_reasoning_finished(
            self._agent.event_bus,
            run_id=self._agent._current_run_id,
            state=self._agent.state,
            role=self._role,
            reasoning=full_text,
        )


@dataclass(slots=True)
class AgentConfig:
    model: ModelSource
    runtime: RuntimeOptions = field(
        default_factory=lambda: RuntimeOptions.auto().with_ctx(8192)
    )
    sandbox_policy: SandboxPolicy = field(
        default_factory=lambda: SandboxPolicy.default()
    )
    name: str = "Agent"
    system_prompt_addition: str = "You are a helpful AI assistant."
    tools: list[Callable] = field(default_factory=list)
    builtin_tools: bool = True
    memory: bool = False
    max_iterations: int = 10
    max_replan: int = 3
    max_consecutive_same_tool_call = 2

    @classmethod
    def from_path(cls, path: str, **kwargs):
        return cls(ModelSource(path), **kwargs)

class Agent(LlamaRuntime):
    def __init__(
        self,
        config: AgentConfig,
    ):
        super().__init__(config.model, config.runtime)
        self._config = config
        self.event_bus: EventBus = EventBus()

        self._config.tools =make_builtin_tools(self,self._config.sandbox_policy)
        self._extractor = TaskExtractor(self)
        self._planner = Planner(
            self, config.tools, reasoning_hook=ReasoningHook(self, "planner")  # pyright: ignore[reportCallIssue]
        )
        self._validator = Validator(self)
        self._synth = Synthesizer(self)
        # Note: self.state/self._current_run_id aren't set until run() starts;
        # ReasoningHook reads them lazily at call time so construction order
        # here doesn't matter. To get the same live "thinking..." visibility
        # for extractor/validator/synth, add a `reasoning_hook` param to
        # their constructors the same way Planner does and pass
        # ReasoningHook(self, "<role>") here.

        if len(config.tools) != 0:
            logger.debug(f"Following Tools are used: {config.tools}")
        else:
            logger.warning(" Attention no tools provided.")

        self._tool_map = self._build_tool_map(config.tools)

    def _build_tool_map(self, tools: list[Callable]):
        return {
            getattr(tool, "__name__", tool.__class__.__name__).lower(): tool
            for tool in tools
        }
    
    def _resolve_tool_name(self, name: str) -> str | None:
        """
        Resolve a planner-provided tool name to a registered tool-map key.

        Tries, in order: exact match -> known alias (TOOL_ALIASES) -> fuzzy
        match against registered names. The fuzzy step exists because grammar-
        constrained decoding guarantees valid JSON syntax, not that the "tool"
        string exactly matches a registered name — models reach for the
        colloquial name ("bash") over the registered one ("run_bash") more
        often than you'd expect. Returns None if nothing reasonable matches.
        """
        if not self._tool_map:
            return None
        if name in self._tool_map:
            return name

        alias = TOOL_ALIASES.get(name)
        if alias and alias in self._tool_map:
            return alias

        close = difflib.get_close_matches(
            name, self._tool_map.keys(), n=1, cutoff=0.6
        )
        return close[0] if close else None


    def _stop_condition(self) -> bool:
        if self.state.is_done:
            return True

        if self.state.step_index >= self._config.max_iterations:
            return True

        return False

    async def _extract_intent_goals(self, query: str) -> ExtractedTask:
        return await self._extractor.run(query)

    async def plan(self, query: str):
        return await self._planner.run(query)

    async def validate(self, query: str):
        return await self._validator.run(query)

    async def synth(self, query: str):
        return await self._synth.run(query)

    def build_context(self):
        """
        Build the LLM context and the tool shortlist.
        Returns:
            full_context, base_context, tools, artifacts
        """
        st = self.state

        base = st.compact_prompt()  # pyright: ignore[reportArgumentType]

        return base

    def _save_get_tool(self,plan:Plan):
        """
        Get the tool and its input from Plan.
        Register Error with correction hint when tool not found directly.

        :returns: The Tool (Callable) and its input (dict) and the raw tool name
        """
        _raw_name = str(plan.tool).lower().strip()

        _tool_map = self._tool_map
        if not _tool_map or not _raw_name in _tool_map:
            _res_name = self._resolve_tool_name(_raw_name)
            self.state.remember_error(f"unknown_tool:{_raw_name} -> might be {_res_name}")
            return None, {}, _raw_name
    
        _tool_inp = plan.input if isinstance(plan.input, dict) else {}
        _tool = _tool_map.get(_raw_name)

        return _tool, _tool_inp, _raw_name

    async def act(self, plan: Plan) -> ToolResult:
        """Execute the plan step produced by the planner."""        
        _tool, _tool_inp, _raw_tool_name = self._save_get_tool(plan) 

        if _tool is None:
            emit_tool_error(
                self.event_bus,
                run_id=self._current_run_id,
                state=self.state,
                tool_name=_raw_tool_name,
                exc=ValueError(f"unknown tool: {_raw_tool_name}"),
            )  # type: ignore[attr-defined]
            return ToolResult(success=False, summary=f"unknown tool: {_raw_tool_name}")

        try:
            result = await _tool(**_tool_inp)
            tool_result_validator(result, _raw_tool_name)
        except Exception as exc:
            self.state.remember_error(
                f"tool_failed:{_raw_tool_name} | {type(exc).__name__}: {exc}"
            )
            emit_tool_error(
                self.event_bus,
                run_id=self._current_run_id,  # type: ignore[attr-defined]
                state=self.state,
                tool_name=_raw_tool_name,
                exc=exc,
            )
            return ToolResult(
                success=False,
                summary=f"tool {_raw_tool_name} raised {type(exc).__name__}: {exc}",
            )

        self.state.remember_fact(f"tool_success:{_raw_tool_name} | {_tool_inp}")

        if getattr(result, "summary", None) and getattr(result, "data", None):
            self.state.remember_result(
                f"tool_result:{_raw_tool_name} | {result.summary} | {result.data}"
            )
        elif getattr(result, "summary", None):
            self.state.remember_result(f"tool_result:{_raw_tool_name} | {result.summary}")

        arts = list(getattr(result, "artifacts", []) or [])
        if arts:
            self.state.extend_artifacts(arts)
            self.state.remember(
                f"artifacts_{self.state.step_index}",
                [a.name for a in arts],
            )

        return result

    async def _synth_helper(self, run_id: str) -> str:
        emit_synthesis_started(self.event_bus, run_id=run_id, state=self.state)
        synth_output = await self.synth(self.state.compact_prompt())
        synth_output = (
            f"{synth_output.response} | References: {synth_output.references}"
        )
        self.state.remember_fact(f"synthesized_output | step={self.state.step_index}")
        emit_synthesis_finished(
            self.event_bus,
            run_id=run_id,
            state=self.state,
            output=synth_output,
        )
        return synth_output

    async def run(self, query: str):
        # Init local vars
        run_id = uuid.uuid4().hex
        self._current_run_id = run_id  # used by act() for tool errors
        event_bus = self.event_bus
        final_output: str | None = None

        emit_agent_started(
            event_bus,
            run_id=run_id,
            query=query,
            tools=list(self._tool_map.keys()),
        )

        try:
            # extract intent and goals
            i_g = await self._extract_intent_goals(query)
            self.state = build_initial_state(query)
            self.state.intent = i_g.intent
            self.state.goals = i_g.goals

            emit_intent_extracted(event_bus, run_id=run_id, state=self.state)
            emit_state_updated(
                event_bus, run_id=run_id, state=self.state, note="intent extracted"
            )

            while not self._stop_condition():
                emit_step_started(event_bus, run_id=run_id, state=self.state)

                # 1. build context of that exe round
                context = self.build_context()

                emit_context_built(
                    event_bus,
                    run_id=run_id,
                    state=self.state,
                    context=context,
                    tools=list(self._tool_map.keys()),
                )

                emit_thinking_started(event_bus, run_id=run_id, state=self.state)
                _plan = await self.plan(context)
                self.state.remember_fact(
                    f"plan:{_plan.tool}({_plan.input}) | {_plan.reason}"
                )

                self.state.set_status("planned")
                emit_action_planned(
                    event_bus, run_id=run_id, state=self.state, action=_plan
                )
                emit_state_updated(
                    event_bus, run_id=run_id, state=self.state, note="plan created"
                )

                if "finish" in str(_plan.tool).lower():
                    self.state.set_status("done")
                    final_output = await self._synth_helper(run_id)
                    emit_agent_done(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        output=final_output,
                    )
                    emit_state_updated(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        note="finished via finish tool",
                    )
                    return self.state

                emit_action_started(
                    event_bus, run_id=run_id, state=self.state, action=_plan
                )
                t1 = time.time()

                result = await self.act(_plan)

                emit_action_finished(
                    event_bus,
                    run_id=run_id,
                    state=self.state,
                    result=result,
                    t0=t1,
                )

                emit_validation_started(event_bus, run_id=run_id, state=self.state)
                t2 = time.time()

                validation = await self.validate(
                    build_validator_prompt(self.state, result, _plan)
                )
                self.state.remember_fact(
                    f"validation:{validation.status} | {validation.reason} | {validation.missing}"
                )
                emit_validation_finished(
                    event_bus,
                    run_id=run_id,
                    state=self.state,
                    validation=validation,
                    t2=t2,
                )

                if validation.status == "done":
                    self.state.set_status("present")
                    emit_state_updated(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        note="validation done",
                    )
                    final_output = await self._synth_helper(run_id)
                    self.state.set_status("done")
                    emit_state_updated(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        note="synthesis complete",
                    )

                elif validation.status == "present":
                    self.state.set_status("present")
                    emit_state_updated(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        note="validation requested presentable output",
                    )
                    await self._synth_helper(run_id)
                    self.state.advance("presented what was done till now")
                    self.state.set_status("working")
                    emit_state_updated(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        note="continued after presenting",
                    )

                elif validation.status == "replan":
                    self.state.advance("Need replanning old approach failed")
                    self.state.set_status("replan")
                    if validation.missing:
                        self.state.open_questions = list(validation.missing)
                    emit_state_updated(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        note="replan requested",
                    )

                else:
                    self.state.advance()
                    self.state.set_status("working")
                    emit_state_updated(
                        event_bus,
                        run_id=run_id,
                        state=self.state,
                        note="continued working",
                    )

            emit_agent_done(
                event_bus, run_id=run_id, state=self.state, output=final_output
            )
            emit_state_updated(
                event_bus, run_id=run_id, state=self.state, note="stopped by condition"
            )
            return self.state

        except Exception as exc:
            if getattr(self, "state", None) is not None:
                self.state.error = f"{type(exc).__name__}: {exc}"
                self.state.set_status("error")
            emit_agent_error(
                event_bus,
                run_id=run_id,
                state=getattr(self, "state", None),
                query=query,
                exc=exc,
            )
            raise


# TODO: State transitions and updates need to be checked and updated
# TODO: Synthesized answer needs to event handled -> update events