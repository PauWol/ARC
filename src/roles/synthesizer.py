from dataclasses import dataclass
from src.roles.base import BaseRole

ANSWER_SYNTHESIZER_PROMPT = """
ROLE: answer synthesizer and presenter

TASK:
Generate a clear, accurate, user-facing response using the provided information.
Transform collected information into the best possible final response for the user.

RULES:
- Use only information present in the input
- Synthesize information into a coherent answer
- Present information clearly and concisely
- Combine related findings when appropriate
- Preserve important facts, results, and constraints
- Do not invent facts or unsupported conclusions
- Do not expose internal reasoning, memory, plans, or agent state
- Omit implementation details unless relevant to the user
- Prefer direct answers over process descriptions
- Adapt structure and detail to the user's request
"""


@dataclass(slots=True)
class SynthesizedAnswer:
    response: str
    references: list[str]


class Synthesizer(BaseRole[SynthesizedAnswer]):
    output_schema = SynthesizedAnswer
    system_prompt = ANSWER_SYNTHESIZER_PROMPT


def build_synth_prompt():
    pass
