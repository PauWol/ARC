from dataclasses import dataclass, field
import difflib
from pathlib import Path
from typing import Callable
import logging

from src.agent.llama_runtime import LlamaRuntime

from src.agent.memory import Session
from src.agent.roles import (
    Extractor,
    Planner,
    Plan,
    ValidationResult,
    Validator,
    build_validator_prompt,
    Synthesizer,
)
from src.agent.schema import ToolResult
from src.tools.utils import tool_result_validator

from src.agent.llama_runtime import ModelSource, RuntimeOptions
from src.agent.policy import SandboxPolicy
from src.tools.builtin import make_builtin_tools

from src.agent.events import EventBus

logger = logging.getLogger("agent")


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

        self._config.tools = make_builtin_tools(self, self._config.sandbox_policy)
        self._extractor = Extractor(self)
        self._planner = Planner(
            self,
            config.tools,
        )
        self._validator = Validator(self)
        self._synth = Synthesizer(self)
        self._session: Session | None = None

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

        close = difflib.get_close_matches(name, self._tool_map.keys(), n=1, cutoff=0.6)
        return close[0] if close else None

    def _stop_condition(self) -> bool:
        if not self.session:
            return False

        if self.session.is_done:
            return True

        if self.session.step_index >= self._config.max_iterations:
            return True

        return False

    async def plan(self, query: str):
        return await self._planner.run(query)

    async def validate(self, query: str):
        return await self._validator.run(query)

    async def synth(self, query: str):
        return await self._synth.run(query)

    def _save_get_tool(self, plan: Plan):
        """
        Get the tool and its input from Plan.
        Register Error with correction hint when tool not found directly.

        :returns: The Tool (Callable) and its input (dict) and the raw tool name
        """
        _raw_name = str(plan.tool).lower().strip()

        _tool_map = self._tool_map
        if not _tool_map or not _raw_name in _tool_map:
            _res_name = self._resolve_tool_name(_raw_name)
            self.state.remember_error(
                f"unknown_tool:{_raw_name} -> might be {_res_name}"
            )
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
            self.state.remember_result(
                f"tool_result:{_raw_tool_name} | {result.summary}"
            )

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

    async def validator(self, validation_result: ValidationResult):
        if validation_result.status == "done":
            self.session.set_state.present
            # TODO Insert actual present method
            self.session.set_state.done

            if self.session.artifacts:
                for a in self.session.artifacts:
                    if not a.path:
                        continue

                    _path = Path(a.path)

                    if _path.exists():
                        continue

                    if a.content:
                        _path.write_text(a.content, "utf-8")

        elif validation_result.status == "present":
            self.session.set_state.present
            # TODO Insert actual present method
            self.session.set_state.working

        elif validation_result.status == "replan":
            self.session.set_state.replan
            if validation_result.missing:
                # TODO: Insert missing working memory
                pass
        else:
            self.session.set_state.working

    async def run(self, query: str):
        self.session = await Session.new(query, self._extractor)

        while not self._stop_condition():
            # ---------- Phase -----------

            # 1. build context of that exe round
            context = self.build_context()

            _plan = await self.plan(context)
            self.state.remember_fact(
                f"plan:{_plan.tool}({_plan.input}) | {_plan.reason}"
            )

            self.session.set_state.planned

            result = await self.act(_plan)

            validation = await self.validate(
                build_validator_prompt(self.state, result, _plan)
            )
            self.state.remember_fact(
                f"validation:{validation.status} | {validation.reason} | {validation.missing}"
            )

        return self.session
