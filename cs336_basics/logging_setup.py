"""Configure console, text-file and JSON-message logging for a named application."""

import logging
from pathlib import Path


def _close_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def setup_logging(
    *,
    log_file: Path | None,
    metrics_log_file: Path | None,
    silent: bool,
    logger_name: str = "cs336_basics",
) -> None:
    """Configure a logger and its children"""

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    _close_handlers(logger)

    metrics_logger = logging.getLogger(f"{logger_name}.metrics")
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.propagate = False
    _close_handlers(metrics_logger)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )

    if not silent:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if metrics_log_file is not None:
        metrics_log_file.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(metrics_log_file, mode="w", encoding="utf-8")
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(logging.Formatter("%(message)s"))
        metrics_logger.addHandler(json_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    if not metrics_logger.handlers:
        metrics_logger.addHandler(logging.NullHandler())
