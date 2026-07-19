import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


from src.constants import (
    LOG_BACKUP_COUNT,
    LOG_LEVEL,
    LOG_FILE,
    LOG_CONSOLE,
    LOG_JSON,
    LOG_MAX_BYTES,
    LOG_ROTATE,
)


class LoggingConfig:
    level: str = LOG_LEVEL  # pyright: ignore[reportAssignmentType]
    file: str = LOG_FILE  # pyright: ignore[reportAssignmentType]
    console: bool = LOG_CONSOLE  # pyright: ignore[reportAssignmentType]
    json: bool = LOG_JSON  # pyright: ignore[reportAssignmentType]
    rotate: bool = LOG_ROTATE  # pyright: ignore[reportAssignmentType]
    max_bytes: int = int(LOG_MAX_BYTES)  # pyright: ignore[reportArgumentType]
    backup_count: int = int(LOG_BACKUP_COUNT)  # pyright: ignore[reportArgumentType]


def setup_logging(cfg: LoggingConfig = LoggingConfig()) -> logging.Logger:
    logger = logging.getLogger("agent")
    logger.setLevel(cfg.level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

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
