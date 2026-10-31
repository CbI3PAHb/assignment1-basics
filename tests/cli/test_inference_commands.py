"""Removed serving commands cannot load configs or generation assets."""

import sys

import pytest

from cs336_basics import cli


@pytest.mark.parametrize("mode", ["mapper", "server"])
@pytest.mark.parametrize("action", ["run", "config"])
def test_removed_inference_command_is_rejected_before_config_loading(mode, action, tmp_path, monkeypatch, capsys):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Removed inference commands must not load a configuration")

    monkeypatch.setattr(cli, "load_config", unexpected_load)
    monkeypatch.setitem(sys.modules, "cs336_basics.generation", None)

    with pytest.raises(SystemExit) as caught:
        cli.main(["inference", mode, action, "--config", str(tmp_path / "missing.yaml")])

    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert captured.out == ""
    assert "invalid choice: 'inference'" in captured.err
    assert sys.modules["cs336_basics.generation"] is None
    assert list(tmp_path.iterdir()) == []
