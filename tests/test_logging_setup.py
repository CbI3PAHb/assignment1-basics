import io
import logging

import pytest

from cs336_basics.logging_setup import setup_logging


@pytest.fixture
def logger_name(monkeypatch):
    monkeypatch.delenv("CS336_LOG_LEVEL", raising=False)
    name = "cs336_basics.tests.logging_setup"
    logger = logging.getLogger(name)
    original_level, original_propagate = logger.level, logger.propagate
    yield name
    setup_logging(logger_name=name, silent=True)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_messages_are_routed_once_to_their_streams(
    logger_name, capsys, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    setup_logging(logger_name=logger_name)
    child = logging.getLogger(f"{logger_name}.transformers")
    child.debug("hidden debug details")
    child.info("training started")
    child.warning("warning message")
    child.error("error message")
    child.critical("critical message")

    output = capsys.readouterr()
    assert output.out.count("training started") == 1
    assert "hidden debug details" not in output.out + output.err
    assert "training started" not in output.err
    for message in ("warning message", "error message", "critical message"):
        assert output.err.count(message) == 1
        assert message not in output.out
    assert list(tmp_path.iterdir()) == []


def test_debug_level_can_be_enabled_with_environment(logger_name, monkeypatch, capsys):
    monkeypatch.setenv("CS336_LOG_LEVEL", "debug")
    setup_logging(logger_name=logger_name)
    logging.getLogger(f"{logger_name}.trainer").debug("debug details")

    output = capsys.readouterr()
    assert output.out.count("debug details") == 1
    assert output.err == ""


def test_reconfiguration_closes_owned_handlers_without_duplicate_output(
    logger_name, capsys
):
    setup_logging(logger_name=logger_name)
    logger = logging.getLogger(logger_name)
    previous_handlers = logger.handlers[:]
    setup_logging(logger_name=logger_name)

    assert all(
        handler not in logger.handlers and handler._closed
        for handler in previous_handlers
    )
    logger.info("one message")
    assert capsys.readouterr().out.count("one message") == 1


def test_silent_disables_both_streams_and_creates_no_files(
    tmp_path, logger_name, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    setup_logging(logger_name=logger_name, silent=True)
    logger = logging.getLogger(f"{logger_name}.trainer")
    logger.info("hidden info")
    logger.error("hidden error")

    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert list(tmp_path.iterdir()) == []


def test_configuring_one_logger_scope_preserves_another(logger_name, capsys):
    names = [f"{logger_name}.bpe", f"{logger_name}.training"]
    try:
        for name in names:
            setup_logging(logger_name=name)
        setup_logging(logger_name=names[0], silent=True)
        logging.getLogger(f"{names[0]}.trainer").info("hidden bpe")
        logging.getLogger(f"{names[1]}.trainer").info("training continues")

        output = capsys.readouterr()
        assert output.out.count("training continues") == 1
        assert "hidden bpe" not in output.out + output.err
    finally:
        for name in names:
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True


def test_root_setup_preserves_host_handlers_and_routes_other_packages(
    monkeypatch, capsys
):
    monkeypatch.delenv("CS336_LOG_LEVEL", raising=False)
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    host_stream = io.StringIO()
    host_handler = logging.StreamHandler(host_stream)
    root.addHandler(host_handler)
    child = logging.getLogger("logging_setup_test_external_package")
    try:
        setup_logging()
        setup_logging()
        assert host_handler in root.handlers
        assert not host_handler._closed
        child.info("external package info")
        child.warning("external package warning")

        output = capsys.readouterr()
        assert output.out.count("external package info") == 1
        assert output.err.count("external package warning") == 1
        assert "external package warning" not in output.out
        assert "external package info" in host_stream.getvalue()
    finally:
        for handler in root.handlers[:]:
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_invalid_environment_level_does_not_modify_handlers(logger_name, monkeypatch):
    setup_logging(logger_name=logger_name)
    logger = logging.getLogger(logger_name)
    original_handlers = logger.handlers[:]
    monkeypatch.setenv("CS336_LOG_LEVEL", "invalid")

    with pytest.raises(ValueError, match="CS336_LOG_LEVEL"):
        setup_logging(logger_name=logger_name)

    assert logger.handlers == original_handlers
    monkeypatch.delenv("CS336_LOG_LEVEL")
