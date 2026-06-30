from dataclasses import dataclass, field
import time
from typing import Callable
import uuid

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
    system_prompt: str = "You are a helpful AI assistant."
    tools: list[Callable] = field(default_factory=list)
    memory: bool = False
    max_iterations: int = 10
    max_replan: int = 3
    max_consecutive_same_tool_call = 2

    @classmethod
    def from_path(cls, path: str, **kwargs):
        return cls(ModelSource(path), **kwargs)

    @classmethod
    def from_path_with_default_tools(
        cls, path: str, sandbox_policy=SandboxPolicy.default(), **kwargs
    ):
        return cls(
            ModelSource(path),
            sandbox_policy=sandbox_policy,
            tools=make_builtin_tools(sandbox_policy),
            **kwargs,
        )


class Agent(LlamaRuntime):
    def __init__(
        self,
        config: AgentConfig,
    ):
        super().__init__(config.model, config.runtime)
        self._config = config
        self.event_bus: EventBus = EventBus()

        self._extractor = TaskExtractor(self)
        self._planner = Planner(self, config.tools)
        self._validator = Validator(self)
        self._synth = Synthesizer(self)

        if len(config.tools) != 0:
            print(f"[DEBUG] Following Tools are used: {config.tools}")
        else:
            print("[DEBUG] Attention no tools provided.")

        self._tool_map = self._build_tool_map(config.tools)

        print(self._tool_map)

    def _build_tool_map(self, tools: list[Callable]):
        return {
            getattr(tool, "__name__", tool.__class__.__name__).lower(): tool
            for tool in tools
        }

    def _is_tool(self, name: str):
        if not self._tool_map:
            return False

        return name in self._tool_map.keys()

    def _get_tool(self, name: str):
        if self._tool_map:
            return self._tool_map.get(name)
        return None

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

    async def act(self, plan: Plan, state: AgentState) -> ToolResult:
        """Execute the plan step produced by the planner."""
        tool_name = str(plan.tool).lower().strip()
        tool_input = plan.input if isinstance(plan.input, dict) else {}
        tool = self._get_tool(tool_name)

        print(f"{tool_name}({tool_input})")

        if tool is None:
            error = ValueError(f"unknown tool: {tool_name}")
            state.remember_error(f"unknown_tool:{tool_name}")
            emit_tool_error(
                self.event_bus,
                run_id=self._current_run_id,
                state=state,
                tool_name=tool_name,
                exc=error,
            )  # type: ignore[attr-defined]
            return ToolResult(success=False, summary=f"unknown tool: {tool_name}")

        try:
            result = await tool(**tool_input)
            tool_result_validator(result, tool_name)
        except Exception as exc:
            state.remember_error(
                f"tool_failed:{tool_name} | {type(exc).__name__}: {exc}"
            )
            emit_tool_error(
                self.event_bus,
                run_id=self._current_run_id,  # type: ignore[attr-defined]
                state=state,
                tool_name=tool_name,
                exc=exc,
            )
            return ToolResult(
                success=False,
                summary=f"tool {tool_name} raised {type(exc).__name__}: {exc}",
            )

        state.remember_fact(f"tool_success:{tool_name} | {tool_input}")

        if getattr(result, "summary", None) and getattr(result, "data", None):
            state.remember_result(
                f"tool_result:{tool_name} | {result.summary} | {result.data}"
            )
        elif getattr(result, "summary", None):
            state.remember_result(f"tool_result:{tool_name} | {result.summary}")

        arts = list(getattr(result, "artifacts", []) or [])
        if arts:
            state.extend_artifacts(arts)
            state.remember(
                f"artifacts_{state.step_index}",
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
        print(synth_output)
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

                result = await self.act(_plan, self.state)

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
