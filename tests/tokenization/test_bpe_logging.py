import logging
from pathlib import Path

from cs336_basics.tokenization.bpe.cli import setup_logging
from cs336_basics.tokenization.bpe import trainer as trainer_module


def test_setup_logging_is_idempotent_for_child_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "bpe.log"
    message = "unique trainer message"

    try:
        setup_logging(
            log_file=log_file,
            json_log_file=None,
            silent=True,
        )
        setup_logging(
            log_file=log_file,
            json_log_file=None,
            silent=True,
        )

        trainer_module.logger.info(message)

        contents = log_file.read_text(encoding="utf-8")
        assert contents.count(message) == 1
    finally:
        setup_logging(
            log_file=None,
            json_log_file=None,
            silent=True,
        )
