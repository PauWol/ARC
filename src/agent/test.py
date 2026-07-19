"""
Example: streamed chat + tool calling in one loop, using llama_runtime.py.

Two-phase pattern per turn, because llama-cpp-python's chatml-function-calling
handler cannot stream while tool_choice="auto" (a permanent upstream
limitation, not a bug — see run_turn()'s docstring below):

  1. Non-streamed "decide" call with tool_choice="auto" — figures out
     whether a tool is needed. Short generation, so streaming loss is minor.
  2. Streamed final answer with tool_choice="none" — no more auto-decision
     left to make, so this streams normally.
"""

import json

from llama_runtime import Conversation, LlamaRuntime, RuntimeOptions, GEN_GREEDY

# ── 1. Define your tools exactly like OpenAI's tools= schema ────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def get_weather(location: str) -> str:
    """Your real implementation would call an API here."""
    return f"18C and cloudy in {location}"


TOOL_IMPL = {"get_weather": get_weather}


# ── 2. Load the model ────────────────────────────────────────────────────────

runtime = LlamaRuntime.from_path(
    "/home/paul/Documents/Programming/Python/ARC/models/Qwen2.5.1-Coder-7B-Instruct-Q4_K_M.gguf",
    RuntimeOptions.auto(n_ctx=8192, chat_format="chatml-function-calling"),
)
runtime.load()


# ── 3. One conversation, streamed, with tool calls handled inline ──────────


def run_turn(convo: Conversation, user_text: str) -> None:
    """
    Run one user turn in two phases, working around a llama-cpp-python
    limitation: chatml-function-calling's tool_choice="auto" cannot stream
    (it raises ValueError("Automatic streaming tool choice is not
    supported") — this is permanent upstream behavior, not a bug in this
    code, see https://github.com/abetlen/llama-cpp-python/discussions/1615).

    Phase 1 (non-streamed): let the model decide whether to call a tool.
    This generation is normally short, so the streaming loss here is minor.
    Phase 2 (streamed): once tool_choice is no longer "auto" — either
    because a tool call is being executed and fed back, or because the
    model already answered directly — the final answer streams normally.
    """
    print(f"\nUser: {user_text}")

    decision = convo.send(user_text, tools=TOOLS, tool_choice="auto")
    message = decision["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        # No tool needed — this was already the full answer, non-streamed.
        print("Assistant:", message["content"])
        return

    # Execute every requested tool call and feed the results back.
    results = []
    for call in tool_calls:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        args = json.loads(args) if isinstance(args, str) else args

        print(f"  [tool call] {name}({args})")
        output = TOOL_IMPL[name](**args)

        results.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or name,
                "content": str(output),
            }
        )
    convo.messages.extend(results)

    # Now stream the final answer. tool_choice="none" keeps the same
    # chat_format handler happy with streaming since there's no more
    # auto-decision left to make.
    print("Assistant: ", end="", flush=True)
    pieces: list[str] = []
    for chunk in convo.runtime.stream_chat(
        convo.messages, reset=False, tools=TOOLS, tool_choice="none", config=GEN_GREEDY
    ):
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            print(delta["content"], end="", flush=True)
            pieces.append(delta["content"])
    print()
    convo.messages.append({"role": "assistant", "content": "".join(pieces)})


# ── 4. Drive it ──────────────────────────────────────────────────────────────

convo = Conversation(runtime, system="You are a helpful assistant with tool access.")

run_turn(convo, "What's the weather in Berlin right now?")
run_turn(convo, "And how does that compare to Rome?")

runtime.close()

# NOTE: run_turn() deliberately does NOT use Conversation.send_stream() for
# the tool-decision phase — chatml-function-calling's tool_choice="auto"
# cannot stream (see the run_turn docstring). send_stream() is still useful
# on its own for plain streamed turns where you already know no tool
# decision is being made (tool_choice="none", or tools omitted entirely).
