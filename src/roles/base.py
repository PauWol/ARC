import json
import re
from typing import TypeVar, Generic, Any, get_args, get_origin, get_type_hints
from dataclasses import fields
from typing import cast
from src.llama_runtime import LlamaRuntime
from src.assets import json_grammar
from src.tools import ToolPromptGenerator

T = TypeVar("T")


class BaseRole(Generic[T]):
    system_prompt: str
    output_schema: type[T]

    def __init__(
        self,
        runtime: LlamaRuntime,
        tokens: int = 512,
        temperature: float = 0.0,
        tools: ToolPromptGenerator | None = None,
    ) -> None:
        self.runtime: LlamaRuntime = runtime
        self._temp: float = temperature
        self._tokens: int = tokens

        self._tool_gen = tools if tools is not None else ToolPromptGenerator.empty()

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
            # Graceful fallback: return raw f.type values (works when no future
            # import is in play).
            return {f.name: f.type for f in fields(schema_cls)}

    def _render_output_schema(self) -> str:
        """
        Converts output_schema dataclass into a JSON-style instruction block
        for the system prompt.
        """
        schema_cls = cast(type, self.output_schema)
        hints = self._resolved_hints()  # ← FIX: use resolved types, not f.type

        schema = {}
        for f in fields(schema_cls):
            field_type = hints.get(f.name, f.type)

            # basic type mapping
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
        - No surrounding textS

        ENFORCEMENT:
        If unsure → still output best-effort valid JSON
        If schema conflicts → follow EXPECTED_SCHEMA strictly
        """
        return p

    def parse_content(self, content: str) -> T:
        data = json.loads(content)

        schema_cls = cast(type, self.output_schema)
        schema_fields = {f.name for f in fields(schema_cls)}

        # FIX: only forward keys that are actually present in the LLM response.
        # The old dict-comprehension used data.get(k), which passes None for every
        # missing key and silently overrides dataclass field defaults — e.g.
        # Thought(tool=None) instead of Thought(tool=""), causing registry.get(None)
        # to return None and the "unknown tool: None" error.
        kwargs = {k: data[k] for k in schema_fields if k in data}

        return self.output_schema(**kwargs)

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
        hints = self._resolved_hints()  # ← FIX: resolve string annotations

        # try full JSON first
        try:
            data = json.loads(partial)
            for f in fields(schema_cls):
                if f.name in data:
                    result[f.name] = data[f.name]

            return self.output_schema(**result)  # type: ignore
        except json.JSONDecodeError:
            pass

        # fallback regex extraction — now works because hints contains real types
        for f in fields(schema_cls):
            key = f.name
            expected_type = hints.get(f.name, f.type)  # ← FIX

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

    async def run(self, query: str):
        response = await self.runtime.achat(
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {
                    "role": "user",
                    "content": query,
                },
            ],
            grammar=json_grammar(),
            temperature=self.temp,
            top_p=0.1,
            max_tokens=self.tokens,
            reset=True,
        )

        content = response["choices"][0]["message"]["content"]
        ct = self.parse_content(content)
        return ct

    async def stream(self, query: str):
        buffer = ""

        async for chunk in self.runtime.astream_chat(
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": query},
            ],
            grammar=json_grammar(),
            temperature=self.temp,
            top_p=0.1,
            max_tokens=self.tokens,
            reset=True,
        ):
            token = chunk["choices"][0]["delta"]["content"]
            buffer += token

            yield self._partial(buffer)
