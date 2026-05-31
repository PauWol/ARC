# example.py
from src.llama_runtime import ModelSource, RuntimeOptions
from src.agent import Agent

ag = Agent(
    ModelSource(model_path="./models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"),
)


QUERY = "Please make me a python programm tht generates passwords when run it should include a cli with params to set"

ag.run(QUERY)
