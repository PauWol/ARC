from dotenv import load_dotenv
from src.logger import setup_logging
from src.schema import LoggingConfig

load_dotenv()

cfg = LoggingConfig()
setup_logging(cfg)

