"""Configure application logging to stdout and stderr; never open log files."""

import logging
import os
import sys


class _MaxLevelFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= logging.INFO


class _ApplicationStreamHandler(logging.StreamHandler):
    """Mark handlers owned by setup_logging so other handlers are preserved."""


class _ApplicationNullHandler(logging.NullHandler):
    """Prevent logging.lastResort from producing output when silent is enabled."""


def setup_logging(*, silent: bool = False, logger_name: str | None = None) -> None:
    """Configure root logging by default, or an explicitly named logger scope.

    DEBUG/INFO go to stdout; WARNING and above go to stderr. CS336_LOG_LEVEL
    controls the minimum level (INFO by default). Reconfiguration replaces only
    our handlers, preserving handlers installed by the host application.
    """
    level_name = os.getenv("CS336_LOG_LEVEL", "INFO").upper()
    levels = logging.getLevelNamesMapping()
    if level_name not in levels:
        raise ValueError(f"Unknown CS336_LOG_LEVEL: {level_name!r}")

    logger = logging.getLogger(logger_name)
    logger.setLevel(levels[level_name])
    if logger_name is not None:
        logger.propagate = False

    for handler in logger.handlers[:]:
        if isinstance(handler, (_ApplicationStreamHandler, _ApplicationNullHandler)):
            logger.removeHandler(handler)
            handler.close()

    if silent:
        logger.addHandler(_ApplicationNullHandler())
        return

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    stdout_handler = _ApplicationStreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(_MaxLevelFilter())
    logger.addHandler(stdout_handler)

    stderr_handler = _ApplicationStreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
