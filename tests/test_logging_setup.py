import json
import logging

import pytest

from cs336_basics.logging_setup import setup_logging


@pytest.fixture
def logger_name():
    name = "cs336_basics.tests.logging_setup"
    yield name
    setup_logging(logger_name=name, log_file=None, metrics_log_file=None, silent=True)


def test_console_and_file_handle_child_messages_at_their_levels(tmp_path, logger_name, capsys):
    path = tmp_path / "logs" / "train.log"
    setup_logging(logger_name=logger_name, log_file=path, metrics_log_file=None, silent=False)
    child = logging.getLogger(f"{logger_name}.transformers")

    child.debug("debug details")
    child.info("training started")

    stderr = capsys.readouterr().err
    assert "training started" in stderr
    assert "debug details" not in stderr
    contents = path.read_text()
    assert contents.count("training started") == 1
    assert "debug details" in contents


def test_json_reconfiguration_closes_old_handlers_and_can_disable_output(tmp_path, logger_name, capsys):
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    setup_logging(logger_name=logger_name, log_file=None, metrics_log_file=old_path, silent=True)
    metrics = logging.getLogger(f"{logger_name}.metrics")
    old_handlers = metrics.handlers[:]
    metrics.info(json.dumps({"step": 1}))

    setup_logging(logger_name=logger_name, log_file=None, metrics_log_file=new_path, silent=True)
    assert all(handler.stream is None for handler in old_handlers)
    metrics.info(json.dumps({"step": 2}))
    new_handlers = metrics.handlers[:]

    setup_logging(logger_name=logger_name, log_file=None, metrics_log_file=None, silent=True)
    assert all(handler.stream is None for handler in new_handlers)
    metrics.info(json.dumps({"step": 3}))

    assert old_path.read_text().splitlines() == [json.dumps({"step": 1})]
    assert new_path.read_text().splitlines() == [json.dumps({"step": 2})]
    assert capsys.readouterr().err == ""


def test_configuring_one_logger_scope_preserves_another(tmp_path, logger_name):
    names = [f"{logger_name}.bpe", f"{logger_name}.training"]
    paths = [tmp_path / "bpe.log", tmp_path / "training.log"]
    try:
        for name, path in zip(names, paths, strict=True):
            setup_logging(logger_name=name, log_file=path, metrics_log_file=None, silent=True)

        setup_logging(logger_name=names[0], log_file=None, metrics_log_file=None, silent=True)
        logging.getLogger(f"{names[1]}.trainer").info("training continues")

        assert "training continues" in paths[1].read_text()
        assert "training continues" not in paths[0].read_text()
    finally:
        for name in names:
            setup_logging(logger_name=name, log_file=None, metrics_log_file=None, silent=True)
