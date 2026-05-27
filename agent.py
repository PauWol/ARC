from llama_runtime import LlamaRuntime, RuntimeOptions, ModelSource

from memory import build_initial_state, AgentState
from extractor import TaskExtractor, ExtractedTask


AVERAGE_CONTEXT_TOKENS = 200


class Agent(LlamaRuntime):
    def __init__(self, source: ModelSource, options: RuntimeOptions | None = None):
        super().__init__(source, options)

    def _pre_process(self, query) -> str:
        return query

    @staticmethod
    def _stop_condition(state: AgentState) -> bool:

        if state.done:
            return True

        return False

    def _extract_intent_goals(self, query: str) -> ExtractedTask:
        tx = TaskExtractor(self)
        return tx.extract(query)

    def build_context(self):
        """Used to build and return the current important context (tool-filtering, goal, intent etc.)."""
        pass

    def think(self):
        """The LLM planning the next step towards goal."""
        pass

    def act(self):
        """Executing the plan (step) made by the LLM."""
        pass

    def validate(self):
        """Another LLM checking if goal is reached or other constrains are missing ('early' exit)."""
        pass

    def run(self, query: str):
        """The main agent entry point to execute/run a task-query with it."""

        query = self._pre_process(query)

        i_g = self._extract_intent_goals(query)

        state = build_initial_state(query)

        state.intent = i_g.intent
        state.goals = i_g.goals

        print(state.as_dict())

        while not self._stop_condition(state):
            pass
