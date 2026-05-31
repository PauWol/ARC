from src.agent import Agent, ModelSource, RuntimeOptions

REASERCH_AGENT_MODEL = ""

SYSTEM_PROMPT = """"""

class ResearchSubAgent(Agent):
    def __init__(self):
        super().__init__(ModelSource(REASERCH_AGENT_MODEL),RuntimeOptions.auto())
        self.load()
        self.set_system_prompt(SYSTEM_PROMPT)
    
    async def run(self,query: str):
        await self.run(query)
