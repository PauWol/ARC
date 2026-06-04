import time
import traceback
import uuid

from src.llama_runtime import LlamaRuntime, RuntimeOptions, ModelSource

from src.memory import build_initial_state, AgentState
from src.roles import TaskExtractor, ExtractedTask, validate, think
from src.tools.registry import ToolRegistry, ToolResult
from src.tools import default_tools

from src.events import Event, EventBus, EventType, emit


class Agent(LlamaRuntime):
    def __init__(self, source: ModelSource, options: RuntimeOptions | None = None):
        super().__init__(source, options or RuntimeOptions.auto())
        self.registry = ToolRegistry()
        default_tools(self.registry)

        self.event_bus = EventBus()

    @staticmethod
    def _stop_condition(state: AgentState) -> bool:

        if state.done:
            return True

        return False

    def _extract_intent_goals(self, query: str) -> ExtractedTask:
        tx = TaskExtractor(self)
        return tx.extract(query)

    def build_context(self, state: AgentState) -> str:
        """Used to build and return the current important context (tool-filtering, goal, intent etc.)."""
        base = state.compact_prompt()

        query = " ".join(
            x
            for x in [
                state.intent,
                " ".join(state.goals),
                " ".join(state.open_questions),
            ]
            if x
        )

        relevant, tools, other = self.registry.build_tool_context(query, top_k=6)

        artifact_lines = ["artifacts:"]
        for art in self.registry.list_artifacts()[-5:]:
            artifact_lines.append(
                f"- {art.type}: {art.name} | {art.description} | {art.path or 'inline'}"
            )

        artifacts = []
        if not len(artifact_lines) > 1:  # guard from empty only 'artifacts:' in list
            artifact_lines = []
        else:
            artifacts = artifact_lines[1:]

        artifacts.extend(other)

        combined = "\n".join([base, "", relevant, "", "\n".join(artifact_lines)])

        return combined, base, tools, artifacts

    def act(self, action, state: AgentState) -> ToolResult:
        """Executing the plan (step) made by the LLM."""
        tool_name = action["tool"]
        tool_input = action.get("input", {})

        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(success=False, summary=f"unknown tool: {tool_name}")

        try:
            result = tool.func(**tool_input)
        except Exception as exc:
            return ToolResult(
                success=False,
                summary=f"tool {tool_name} raised {type(exc).__name__}: {exc}",
            )

        if isinstance(result, ToolResult):
            for art in result.artifacts:
                self.registry.register_artifact(art)
            if result.artifacts:
                state.remember(
                    f"artifact_{state.step_index}",
                    [a.name for a in result.artifacts],
                )
            return result

        return ToolResult(success=True, summary=str(result))

    def run(self, query: str):
        run_id = uuid.uuid4().hex
        event_bus = self.event_bus
        emit(event_bus, 0, run_id=run_id, query=query)

        f_p_c = 0

        try:
            i_g = self._extract_intent_goals(query)

            state = build_initial_state(query)
            state.intent = i_g.intent
            state.goals = i_g.goals

            emit(event_bus, 1, run_id=run_id, state=state)

            while not self._stop_condition(state):
                emit(event_bus, 2, run_id=run_id, state=state)

                context, base, tools, artifacts = self.build_context(state)

                emit(
                    event_bus,
                    3,
                    run_id=run_id,
                    state=state,
                    context=context,
                    base=base,
                    tools=tools,
                    artifacts=artifacts,
                )

                emit(event_bus, 4, run_id=run_id, state=state)
                t0 = time.time()

                action = think(self, state, context=context)

                emit(event_bus, 5, run_id=run_id, state=state, t0=t0, action=action)
                emit(event_bus, 6, run_id=run_id, state=state, action=action)

                t1 = time.time()

                result = self.act(action, state)

                emit(event_bus, 7, run_id=run_id, state=state, result=result, t1=t1)

                state.remember(f"obs_{state.step_index}", result.summary)

                emit(event_bus, 8, run_id=run_id, state=state)
                t2 = time.time()

                if f_p_c > 3:
                    f = False
                    f_p_c = 0
                else:
                    f = True

                validation, fastPath = validate(
                    self, state, action, result, self.registry.list_artifacts(), False
                )
                if fastPath:
                    f_p_c += 1

                emit(
                    event_bus,
                    9,
                    run_id=run_id,
                    state=state,
                    validation=validation,
                    t2=t2,
                )

                if validation.done or validation.status == "complete":
                    state.done = True
                    state.status = "done"
                elif validation.status == "present":
                    state.status = "present"
                elif validation.status == "replan":
                    state.advance("Need replanning old approach failed")
                    state.status = "new"
                    state.open_questions = validation.missing
                else:
                    state.advance()
                    state.status = "working"

            emit(event_bus, 10, run_id=run_id, state=state)

        except Exception as exc:
            emit(event_bus, 11, run_id=run_id, state=state, exc=exc, query=query)  # type: ignore
            raise

        finally:
            self.registry.close()
