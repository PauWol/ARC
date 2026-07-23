from dataclasses import dataclass, field
import difflib
from pathlib import Path
from typing import Any, Callable
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

from src.agent.events import EventEmitter

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
        self._config.tools = make_builtin_tools(self, self._config.sandbox_policy)
        self._extractor = Extractor(self)
        self._planner = Planner(
            self,
            config.tools,
        )
        self._validator = Validator(self)
        self._synth = Synthesizer(self)
        self._session: Session | None = None
        self.event: EventEmitter

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
        self.event.thinking_started()
        _plan = await self._planner.run(query)
        self.event.thinking_finished(_plan.tool, _plan.reason, _plan.input)
        self.session.set_state.planned
        return _plan

    async def validate(self, query: str):
        validation_result = await self._validator.run(query)

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

    async def synth(self, query: str):
        return await self._synth.run(query)

    async def act(self, plan: Plan) -> ToolResult:
        """Execute the plan step produced by the planner."""

        pass

    async def present(self):
        pass

    async def run(self, query: str):
        self.session = await Session.new(query, self._extractor)
        self.event = EventEmitter.with_agent_started(self.session.id)

        while not self._stop_condition():
            _plan = await self.plan(query)
            result = await self.act(_plan)
            validation = await self.validate(_plan, result)

        return self.session
