"""Command-line entrypoint for viewing a training config or running training."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .training.config.io import dump_experiment, load_experiment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a training action and exactly one YAML config path."""
    parser = argparse.ArgumentParser(prog="cs336-basics")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="View a configuration or run training.")
    train_parser.add_argument("action", choices=("run", "config"))
    train_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="Path to one complete experiment YAML file.",
    )

    args = parser.parse_args(argv)
    if len(args.config) != 1:
        parser.error("--config must be specified exactly once")

    args.config = args.config[0]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Load the experiment config, then print it or run training."""
    args = parse_args(argv)

    config = load_experiment(args.config)
    match args.action:
        case "config":
            print(dump_experiment(config), end="")
        case "run":
            from .training.run import run_training

            run_training(config)


if __name__ == "__main__":
    main()
