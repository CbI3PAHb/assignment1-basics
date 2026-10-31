import argparse
from collections.abc import Sequence
from pathlib import Path

from cs336_basics.logging_setup import setup_logging

from .config import ExperimentConfig, dump_config, load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cs336-basics")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train a model.")
    train_parser.set_defaults(config_cls=ExperimentConfig)
    train_parser.add_argument("action", choices=("run", "config"))
    train_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="Path to one complete YAML config",
    )

    args = parser.parse_args(argv)
    if len(args.config) != 1:
        parser.error("--config must be specified exactly once")
    args.config = args.config[0]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Parse args, load YAML config, execute its program or show the config itself"""
    args = parse_args(argv)
    config = load_config(args.config, args.config_cls)
    setup_logging(silent=config.logging.silent)

    print(dump_config(config), end="")

    if args.action == "config":
        return

    from .training.run import run_training

    run_training(config)


if __name__ == "__main__":
    main()
