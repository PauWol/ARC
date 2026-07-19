"""
Model family profiles.

Solves three problems that a single hardcoded `achat(grammar=json_grammar())`
call can't:

1. THINKING MODELS. DeepSeek-R1(-distill-*), QwQ, and Qwen3/3.5 in thinking
   mode emit a free-form <think>...</think> block before their real answer.
   Forcing a JSON grammar from the very first generated token either blocks
   the model from thinking at all, or breaks generation outright (the
   grammar has no path for "<think>"). These models also want different
   sampling than greedy JSON extraction — DeepSeek-R1 in particular degrades
   noticeably at temperature 0.

2. "AUTO-THINK" FOR NON-THINKING MODELS. Even plain Qwen2.5-instruct
   benefits from a scratch-space reasoning pass before being asked to commit
   to a single structured JSON answer. This should be an opt-in toggle per
   role, not something hardcoded to specific model families.

3. CHAT-TEMPLATE QUIRKS. Some models (Gemma) don't support a "system" role
   at all and need it folded into the first user turn.

The fix used throughout src/roles/base.py is: never run grammar-constrained
decoding in the same generation as free-form reasoning. Always do reasoning
(if enabled) as its own unconstrained call, then start a *fresh*
grammar-constrained call with the reasoning fed back in as context. This
works identically for native thinking models and manually-added CoT, and
never conflicts with GBNF grammars because the constrained call never has to
emit anything but the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern

from src.llama_runtime import GenerationConfig, ModelSource


@dataclass(frozen=True)
class ModelProfile:
    name: str
    match: tuple[Pattern, ...] = ()

    # True for models that natively emit <think>...</think> without being
    # asked (DeepSeek-R1, QwQ, Qwen3-thinking variants).
    is_thinking: bool = False
    think_open: str = "<think>"
    think_close: str = "</think>"

    # Some finetunes only reliably think if the assistant turn is pre-seeded
    # with "<think>\n" via raw completion rather than the chat template.
    # When True, BaseRole._reason() uses runtime.complete() with a manually
    # built prompt instead of achat().
    force_open_think: bool = False

    supports_system_role: bool = True

    # Whether it's safe to apply a JSON grammar to a generation that might
    # also contain this model's native thinking tokens. Always False for
    # is_thinking=True profiles in this codebase, since we never actually
    # try to grammar-constrain a thinking generation — kept as an explicit
    # field so a future backend that *can* do mid-stream grammar switching
    # has somewhere to declare that capability.
    grammar_safe_from_start: bool = True

    default_sampling: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.0, top_p=0.1, top_k=40)
    )
    reasoning_sampling: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.6, top_p=0.95, top_k=40)
    )

    def matches(self, haystack: str) -> bool:
        return any(p.search(haystack) for p in self.match)


PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        name="deepseek-r1",
        match=(
            re.compile(r"deepseek-r1", re.I),
            re.compile(r"deepseek.*distill", re.I),
        ),
        is_thinking=True,
        force_open_think=True,
        grammar_safe_from_start=False,
        # DeepSeek's own model card: temp 0.5-0.7 (0.6 default), never greedy.
        default_sampling=GenerationConfig(temperature=0.6, top_p=0.95, top_k=40),
        reasoning_sampling=GenerationConfig(temperature=0.6, top_p=0.95, top_k=40),
    ),
    ModelProfile(
        name="qwq",
        match=(re.compile(r"qwq", re.I),),
        is_thinking=True,
        grammar_safe_from_start=False,
        default_sampling=GenerationConfig(temperature=0.6, top_p=0.95, top_k=30),
        reasoning_sampling=GenerationConfig(temperature=0.6, top_p=0.95, top_k=30),
    ),
    ModelProfile(
        name="qwen3-thinking",
        match=(
            re.compile(r"qwen3[._-]?5?.*thinking", re.I),
            re.compile(r"qwen.*thinking", re.I),
        ),
        is_thinking=True,
        grammar_safe_from_start=False,
        default_sampling=GenerationConfig(temperature=0.6, top_p=0.95, top_k=20),
        reasoning_sampling=GenerationConfig(temperature=0.6, top_p=0.95, top_k=20),
    ),
    ModelProfile(
        # plain (non-"-thinking"-tagged) Qwen3 / Qwen3.5 builds — support
        # switchable thinking via chat-template hints but default off.
        name="qwen3",
        match=(re.compile(r"qwen3(\.5)?(?!.*thinking)", re.I),),
        is_thinking=False,
        grammar_safe_from_start=True,
        default_sampling=GenerationConfig(temperature=0.0, top_p=0.1, top_k=40),
        reasoning_sampling=GenerationConfig(temperature=0.6, top_p=0.8, top_k=20),
    ),
    ModelProfile(
        name="qwen2.5",
        match=(re.compile(r"qwen2[._-]?5", re.I),),
        is_thinking=False,
        grammar_safe_from_start=True,
        default_sampling=GenerationConfig(temperature=0.0, top_p=0.1, top_k=40),
        reasoning_sampling=GenerationConfig(temperature=0.5, top_p=0.9, top_k=40),
    ),
    ModelProfile(
        name="gemma",
        match=(re.compile(r"gemma", re.I),),
        is_thinking=False,
        supports_system_role=False,
        grammar_safe_from_start=True,
    ),
)

GENERIC = ModelProfile(name="generic")

_BY_NAME = {p.name: p for p in PROFILES}


def detect_profile(source: ModelSource, override: str | None = None) -> ModelProfile:
    """
    Best-effort family detection from model_path / repo_id / filename.

    Pass `override` (or set `ModelSource.family`) to skip detection, e.g.
    when a filename doesn't carry the family in it ("model-final.gguf").
    """
    if override:
        try:
            return _BY_NAME[override]
        except KeyError:
            raise ValueError(
                f"Unknown model profile {override!r}. Known: {sorted(_BY_NAME)}"
            ) from None

    haystack = " ".join(
        filter(None, [source.model_path, source.repo_id, source.filename])
    )
    for p in PROFILES:
        if p.matches(haystack):
            return p
    return GENERIC