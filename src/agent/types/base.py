import json
import re
from typing import (
    TypeVar,
    Generic,
    Any,
    Literal,
    Protocol,
    get_args,
    get_origin,
    get_type_hints,
)
from dataclasses import fields
from typing import cast
from src.agent.llama_runtime import LlamaRuntime
from src.agent.profiles import ModelProfile
from src.assets import json_grammar
from src.tools import ToolPromptGenerator

T = TypeVar("T")

ReasoningMode = Literal["auto", "on", "off"]

REASON_PROMPT_SUFFIX = """
Before answering, think through the problem in plain prose.
Do not output JSON in this step. Do not restate these instructions.
Be concise: a few short sentences or a short numbered list is enough.
"""


class ReasoningSink(Protocol):
    """
    Structural interface a caller (typically Agent) implements to observe a
    role's reasoning pass live, e.g. to stream "thinking..." into a UI.
    All three methods are called only when reasoning is actually enabled
    for this run (see ReasoningMode); no-ops are fine to implement.
    """

    def start(self) -> None: ...
    def chunk(self, text: str) -> None: ...
    def finish(self, full_text: str) -> None: ...


class BaseRole(Generic[T]):
    system_prompt: str
    output_schema: type[T]

    def __init__(
        self,
        runtime: LlamaRuntime,
        tokens: int = 512,
        temperature: float = 0.0,
        tools: ToolPromptGenerator | None = None,
        *,
        reasoning: ReasoningMode = "auto",
        reasoning_tokens: int = 400,
        repair_attempts: int = 1,
        reasoning_hook: ReasoningSink | None = None,
        role_name: str | None = None,
    ) -> None:
        self.runtime: LlamaRuntime = runtime
        self._temp: float = temperature
        self._tokens: int = tokens

        self._tool_gen = tools if tools is not None else ToolPromptGenerator.empty()

        # "auto"  -> reason only if the loaded model is a native thinking model
        # "on"    -> always run an explicit reasoning pass first (CoT for any model)
        # "off"   -> never reason, keep old single-shot grammar behavior
        self._reasoning_mode: ReasoningMode = reasoning
        self._reasoning_tokens = reasoning_tokens
        self._repair_attempts = max(0, repair_attempts)

        # Optional live observer for the reasoning pass — Agent wires this to
        # the event bus so a UI can show "thinking..." as tokens stream in.
        self._reasoning_hook = reasoning_hook
        self.role_name = role_name or type(self).__name__

        # Populated after each run()/stream() call for logging/telemetry.
        self.last_reasoning: str | None = None

    # ── schema rendering (unchanged) ────────────────────────────────────────

    def _resolved_hints(self) -> dict[str, Any]:
        """
        Return resolved type hints for the output schema.

        When a module uses `from __future__ import annotations` (PEP 563), every
        annotation is stored as a *string* at runtime, so `f.type` returns e.g.
        `"str"` instead of the built-in `str`.  Identity checks like
        `field_type is str` then always fail, causing every field to be classified
        as "any" in the schema prompt and breaking the regex fallback in `_partial`.

        `typing.get_type_hints()` evaluates those string annotations back to the
        actual types, so all downstream comparisons work correctly regardless of
        whether the dataclass module uses the future import.
        """
        schema_cls = cast(type, self.output_schema)
        try:
            return get_type_hints(schema_cls)
        except Exception:
            return {f.name: f.type for f in fields(schema_cls)}

    def _render_output_schema(self) -> str:
        schema_cls = cast(type, self.output_schema)
        hints = self._resolved_hints()

        schema = {}
        for f in fields(schema_cls):
            field_type = hints.get(f.name, f.type)

            if field_type is str:
                type_str = "string"
            elif field_type is int:
                type_str = "integer"
            elif field_type is float:
                type_str = "number"
            elif field_type is bool:
                type_str = "boolean"
            elif get_origin(field_type) is list:
                inner = get_args(field_type)[0]
                if inner is str:
                    type_str = "array[string]"
                elif inner is int:
                    type_str = "array[integer]"
                elif inner is float:
                    type_str = "array[number]"
                else:
                    type_str = "array[unknown]"
            elif get_origin(field_type) is dict:
                type_str = "object"
            else:
                type_str = "any"

            schema[f.name] = type_str

        return json.dumps(schema, indent=2)

    def build_system_prompt(self, optional_append: str = "") -> str:
        base = self.system_prompt.strip()
        schema_block = self._render_output_schema()
        tool_block = self._tool_gen.build_prompt()

        p = f"""
        SYSTEM:
        {base}

        {optional_append}

        {tool_block}

        SCHEMA:
        You MUST respond ONLY in valid JSON.

        EXPECTED_SCHEMA:
        {schema_block}

        STRICT_RULES:
        - Output must match schema exactly
        - No extra keys
        - No explanations
        - No markdown
        - No surrounding text

        ENFORCEMENT:
        If unsure → still output best-effort valid JSON
        If schema conflicts → follow EXPECTED_SCHEMA strictly
        """.strip()
        return p

    def build_reasoning_prompt(self, optional_append: str = ""):
        base = self.system_prompt.strip()
        tool_block = self._tool_gen.build_prompt()

        p = f"""
        SYSTEM:
        {base}

        {optional_append}

        {tool_block}

        {REASON_PROMPT_SUFFIX}
        """.strip()
        return p

    def parse_content(self, content: str) -> T:
        data = json.loads(content)
        schema_cls = cast(type, self.output_schema)
        schema_fields = {f.name for f in fields(schema_cls)}
        kwargs = {k: data[k] for k in schema_fields if k in data}
        return self.output_schema(**kwargs)

    def parse_content_safe(self, content: str) -> T:
        """
        parse_content() with a fallback chain: raw JSON -> regex-based
        _partial() extraction. GBNF grammars guarantee syntax, not that every
        required dataclass field is present or that repair via run() has
        already happened — callers that want a repair *retry* (a fresh model
        call) should use run(), which calls this only as the last local step.
        """
        try:
            return self.parse_content(content)
        except json.JSONDecodeError, TypeError, KeyError:
            return self._partial(content)

    @property
    def schema(self):
        schema_cls = cast(type, self.output_schema)
        return {f.name: f.type for f in fields(schema_cls)}

    @property
    def empty_schema(self) -> dict[str, None]:
        schema_cls = cast(type, self.output_schema)
        return {f.name: None for f in fields(schema_cls)}

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, _val: float):
        self._temp = _val

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, _val: int):
        self._tokens = _val

    def _partial(self, partial: str) -> T:
        schema_cls = cast(type, self.output_schema)
        result: dict[str, Any] = {f.name: None for f in fields(schema_cls)}
        hints = self._resolved_hints()

        try:
            data = json.loads(partial)
            for f in fields(schema_cls):
                if f.name in data:
                    result[f.name] = data[f.name]
            return self.output_schema(**result)  # type: ignore
        except json.JSONDecodeError:
            pass

        for f in fields(schema_cls):
            key = f.name
            expected_type = hints.get(f.name, f.type)

            if expected_type is str:
                m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)', partial)
                if m:
                    result[key] = m.group(1)

            elif get_origin(expected_type) is list:
                m = re.search(
                    rf'"{re.escape(key)}"\s*:\s*\[([^\]]*)',
                    partial,
                    re.DOTALL,
                )
                if m:
                    raw = m.group(1)
                    item_type = get_args(expected_type)[0]

                    if item_type is str:
                        result[key] = re.findall(r'"([^"]*)', raw)
                    elif item_type in (int, float):
                        regex = r"-?\d+(?:\.\d+)?" if item_type is float else r"-?\d+"
                        result[key] = [item_type(x) for x in re.findall(regex, raw)]

        return self.output_schema(**result)  # type: ignore

    # ── reasoning / model-family handling ───────────────────────────────────

    def _profile(self) -> ModelProfile:
        return self.runtime.profile

    def _wants_reasoning(self, profile: ModelProfile) -> bool:
        if self._reasoning_mode == "off":
            return False
        if self._reasoning_mode == "on":
            return True
        return profile.is_thinking  # "auto"

    def _chat_messages(
        self, profile: ModelProfile, system: str, user: str
    ) -> list[dict[str, str]]:
        """Fold system into the user turn for models with no system role."""
        if profile.supports_system_role:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        return [{"role": "user", "content": f"{system}\n\n{user}"}]

    async def _reason(self, query: str, profile: ModelProfile) -> str:
        """
        Unconstrained, STREAMED reasoning pass. Never carries a grammar, so
        it never conflicts with a native <think> block or free-form CoT
        text. If a reasoning_hook was provided, tokens are forwarded to it
        live (start() -> chunk()* -> finish()) so a UI can show "thinking..."
        as it happens rather than only after the whole pass completes.
        Returns the reasoning text with any <think>...</think> wrapper
        stripped (the wrapper itself is model-specific chrome, not content).
        """
        reason_system = self.build_reasoning_prompt()
        cfg = profile.reasoning_sampling.merge(max_tokens=self._reasoning_tokens)

        hook = self._reasoning_hook
        if hook is not None:
            hook.start()

        buffer = ""
        try:
            if profile.force_open_think:
                # Some distills only reliably enter thinking mode when the
                # assistant turn is pre-seeded via raw completion rather
                # than going through the chat template's own turn-taking.
                prompt = (
                    f"{reason_system}\n\nUSER:\n{query}\n\n"
                    f"ASSISTANT:\n{profile.think_open}\n"
                )
                async for c in self.runtime.astream_complete(
                    prompt,
                    config=cfg,
                    reset=True,
                    stop=[profile.think_close if profile.is_thinking else "\n\n\n"],
                ):
                    token = c["choices"][0].get("text", "")
                    if token:
                        buffer += token
                        if hook is not None:
                            hook.chunk(token)
            else:
                messages = self._chat_messages(profile, reason_system, query)
                async for c in self.runtime.astream_chat(
                    messages=messages, config=cfg, reset=True
                ):
                    token = c["choices"][0]["delta"].get("content", "")
                    if token:
                        buffer += token
                        if hook is not None:
                            hook.chunk(token)
        finally:
            pass

        # Strip a native <think>...</think> wrapper if present; keep the
        # inner content either way — it's exactly the scratch reasoning we
        # want to hand to the answer pass.
        m = re.search(
            re.escape(profile.think_open) + r"(.*?)" + re.escape(profile.think_close),
            buffer,
            re.DOTALL,
        )
        result = m.group(1).strip() if m else buffer.strip()

        if hook is not None:
            hook.finish(result)

        return result

    async def _answer(
        self, query: str, profile: ModelProfile, reasoning: str | None
    ) -> str:
        """Fresh, grammar-constrained call. Never shares a generation with
        the reasoning pass, so the grammar never has to coexist with
        free-form thinking tokens."""
        append = ""
        if reasoning:
            append = (
                "REASONING (already produced, do not repeat it verbatim, "
                "use it to inform your answer):\n" + reasoning
            )

        system = self.build_system_prompt(append)
        messages = self._chat_messages(profile, system, query)

        cfg = profile.default_sampling.merge(
            max_tokens=self.tokens, temperature=self.temp, top_p=0.1
        )
        resp = await self.runtime.achat(
            messages=messages,
            config=cfg,
            grammar=json_grammar(),
            reset=True,
        )
        return resp["choices"][0]["message"]["content"]

    async def _repair(
        self, query: str, profile: ModelProfile, bad_output: str, error: str
    ) -> str:
        """One extra grammar-constrained call showing the model its own
        invalid output plus the parse error, asking it to fix just the
        JSON. Only used when parse_content_safe's local fallback chain
        (raw parse -> regex partial) still can't produce every schema key."""
        append = (
            f"Your previous output was invalid: {error}\n"
            f"PREVIOUS_OUTPUT:\n{bad_output}\n"
            "Fix it. Output ONLY corrected JSON matching EXPECTED_SCHEMA."
        )
        system = self.build_system_prompt(append)
        messages = self._chat_messages(profile, system, query)
        cfg = profile.default_sampling.merge(max_tokens=self.tokens, temperature=0.0)
        resp = await self.runtime.achat(
            messages=messages, config=cfg, grammar=json_grammar(), reset=True
        )
        return resp["choices"][0]["message"]["content"]

    def _required_fields_present(self, obj: T) -> bool:
        schema_cls = cast(type, self.output_schema)
        for f in fields(schema_cls):
            if f.default is not None and getattr(obj, f.name) is None:
                # Field was declared with a non-None default but got None —
                # a strong signal the key was simply missing from the JSON.
                return False
        return True

    def _validate(self, obj: T) -> str | None:
        """
        Optional semantic validation beyond "does every field have a value".
        A GBNF grammar guarantees syntax, not meaning — e.g. Plan.input is
        typed dict[str, Any], so {} is syntactically valid JSON even when
        the chosen tool needs arguments. Override in a subclass and return
        a short description of what's wrong to trigger the repair pass, or
        None if the output is acceptable. Default: no extra checks.
        """
        return None

    # ── public API ───────────────────────────────────────────────────────

    async def run(self, query: str) -> T:
        profile = self._profile()

        reasoning: str | None = None
        if self._wants_reasoning(profile):
            reasoning = await self._reason(query, profile)
            self.last_reasoning = reasoning

        content = await self._answer(query, profile, reasoning)

        def _check(obj: T) -> str | None:
            if not self._required_fields_present(obj):
                return "missing required field(s)"
            return self._validate(obj)

        try:
            result = self.parse_content_safe(content)
        except Exception as exc:
            result = None
            last_error = str(exc)
        else:
            last_error = _check(result)

        attempts = 0
        while last_error is not None and attempts < self._repair_attempts:
            attempts += 1
            content = await self._repair(query, profile, content, last_error)
            try:
                result = self.parse_content_safe(content)
                last_error = _check(result)
            except Exception as exc:
                last_error = str(exc)

        if result is None:
            # Exhausted repairs — return best-effort partial rather than raise,
            # matching the old behavior's leniency.
            result = self._partial(content)

        return result

    async def stream(self, query: str):
        """
        Streaming variant. If reasoning is enabled, the reasoning pass runs
        to completion first (it's short and not grammar-constrained, so it
        isn't worth streaming to the caller) and only the constrained answer
        pass is streamed, matching the old partial-JSON streaming contract.
        """
        profile = self._profile()

        reasoning: str | None = None
        if self._wants_reasoning(profile):
            reasoning = await self._reason(query, profile)
            self.last_reasoning = reasoning

        append = ""
        if reasoning:
            append = (
                "REASONING (already produced, do not repeat it verbatim, "
                "use it to inform your answer):\n" + reasoning
            )
        system = self.build_system_prompt(append)
        messages = self._chat_messages(profile, system, query)
        cfg = profile.default_sampling.merge(
            max_tokens=self.tokens, temperature=self.temp, top_p=0.1
        )

        buffer = ""
        async for chunk in self.runtime.astream_chat(
            messages=messages,
            config=cfg,
            grammar=json_grammar(),
            reset=True,
        ):
            token = chunk["choices"][0]["delta"].get("content", "")
            buffer += token
            yield self._partial(buffer)
