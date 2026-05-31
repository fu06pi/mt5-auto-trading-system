from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: str = DEFAULT_FORMAT,
) -> logging.Logger:
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(format_string)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    return setup_logging(name=name, level=level)
