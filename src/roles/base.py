import json
from typing import TypeVar, get_type_hints, Generic, Any
from dataclasses import fields
from src.llama_runtime import LlamaRuntime
from src.assets import json_grammar

T = TypeVar("T")


class BaseRole(Generic[T]):
    system_prompt: str  # pyright: ignore[reportUninitializedInstanceVariable]
    output_schema: type[T]  # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(
        self, runtime: LlamaRuntime, tokens: int = 500, temperature: float = 0.0
    ) -> None:
        self.runtime: LlamaRuntime = runtime
        self._temp: float = temperature
        self._tokens: int = tokens

    def parse_content(self, content: str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self.empty_schema

        a = {}
        for i in self.schema.keys():
            a[i] = data.get(i, None)

        return a

    @property
    def schema(self):
        return get_type_hints(self.output_schema)

    @property
    def empty_schema(self) -> dict[str, None]:
        return {f.name: None for f in fields(self.output_schema)}

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

    async def run(self, query: str):
        response = await self.runtime.achat(
            messages=[
                {"role": "system", "content": self.system_prompt.strip()},
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
        return self.parse_content(content)

    async def stream(self, query: str):
        buffer = ""

        async for chunk in self.runtime.astream_chat(
            messages=[
                {"role": "system", "content": self.system_prompt.strip()},
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


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Ex:
        tool: str
        tool_name: str

        test: list[str]

    class Test(BaseRole[Ex]):
        system_prompt = """"""
        output_schema = Ex

    t = Test()
    print(t.schema)
