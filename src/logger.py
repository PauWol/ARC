import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.schema import LoggingConfig

def setup_logging(cfg: LoggingConfig) -> logging.Logger:
    logger = logging.getLogger("agent")
    logger.setLevel(cfg.level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )

    Path(cfg.file).parent.mkdir(parents=True, exist_ok=True)

    if cfg.rotate:
        fh = RotatingFileHandler(
            cfg.file,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
    else:
        fh = logging.FileHandler(cfg.file, encoding="utf-8")

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if cfg.console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger