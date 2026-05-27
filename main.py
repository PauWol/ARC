# example.py
from llama_runtime import ModelSource, RuntimeOptions, LlamaRuntime
from agent import Agent

ag = Agent(
    ModelSource(model_path="./models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"),
    RuntimeOptions(
        n_ctx=6000,
        n_gpu_layers=0,  # raise this for GPU offload
        idle_unload_seconds=120,
        chat_format=None,  # set if your model needs it
    ),
)


QUERY = "Make me a prime-number python program if you are capable make me a good visualization of the output"

ag.run(QUERY)
