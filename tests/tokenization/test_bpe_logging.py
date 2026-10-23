import logging
from unittest.mock import Mock

from cs336_basics.logging_setup import setup_logging
from cs336_basics.tokenization.bpe import cli as bpe_cli
from cs336_basics.tokenization.bpe import trainer as trainer_module


def test_setup_logging_is_idempotent_for_child_logger(capsys, monkeypatch) -> None:
    monkeypatch.delenv("CS336_LOG_LEVEL", raising=False)
    message = "unique trainer message"
    logger = logging.getLogger("cs336_basics.tokenization.bpe")
    original_handlers = logger.handlers[:]
    original_level, original_propagate = logger.level, logger.propagate
    try:
        setup_logging(logger_name="cs336_basics.tokenization.bpe")
        setup_logging(logger_name="cs336_basics.tokenization.bpe")

        trainer_module.logger.info(message)

        output = capsys.readouterr()
        assert output.out.count(message) == 1
        assert output.err == ""
    finally:
        for handler in logger.handlers[:]:
            if handler not in original_handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.handlers[:] = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_bpe_main_configures_stream_logging_without_file_arguments(
    tmp_path, monkeypatch
):
    configure = Mock()
    train = Mock(return_value=({0: b"a"}, []))
    save = Mock()
    monkeypatch.setattr(bpe_cli, "setup_logging", configure)
    monkeypatch.setattr(bpe_cli, "train_bpe", train)
    monkeypatch.setattr(bpe_cli, "save_tokenizer", save)

    bpe_cli.main(
        [
            "--dataset-file-path",
            str(tmp_path / "input.txt"),
            "--save-file-path",
            str(tmp_path / "tokenizer.pkl"),
            "--vocab-size",
            "256",
            "--silent",
        ]
    )

    configure.assert_called_once_with(silent=True)
    train.assert_called_once()
    save.assert_called_once()
