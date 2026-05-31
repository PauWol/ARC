# example.py
from src.llama_runtime import ModelSource, RuntimeOptions
from src.agent import Agent

ag = Agent(
    ModelSource(model_path="./models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"),
    RuntimeOptions(
        n_ctx=6000,
        n_gpu_layers=0,  # raise this for GPU offload
        idle_unload_seconds=120,
        chat_format=None,  # set if your model needs it
    ),
)


QUERY = "What is two plus two times 20 minus pi"

ag.run(QUERY)
