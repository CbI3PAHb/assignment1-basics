from pathlib import Path


def _prepare_output_directory(output_dir: Path, config_yaml: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config.yaml"
    config_created = False
    try:
        with config_path.open("x", encoding="utf-8") as config_file:
            config_created = True
            config_file.write(config_yaml)
    except BaseException:
        # Remove only our new config and directory; preserve unrelated files.
        try:
            if config_created:
                config_path.unlink(missing_ok=True)
            output_dir.rmdir()
        except OSError:
            pass
        raise
