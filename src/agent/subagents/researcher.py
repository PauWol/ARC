from src.agent import Agent, AgentConfig


async def research(query: str, main_conf: AgentConfig):
    conf = AgentConfig(main_conf.model, main_conf.runtime, builtin_tools=False)
    agent = Agent()
    pass
