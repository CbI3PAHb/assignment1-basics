"""Command-line entrypoint for training, mapper inference and serving"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from cs336_basics.logging_setup import setup_logging

from .config.inference import InferenceMapperConfig, InferenceServerConfig
from .config.io import dump_config, load_config
from .config.training import ExperimentConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cs336-basics")
    parser.set_defaults(mode=None)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train a model.")
    inference_parser = commands.add_parser("inference", help="Generate text.")

    modes = inference_parser.add_subparsers(dest="mode", required=True)
    mapper_parser = modes.add_parser("mapper", help="Process known input and exit.")
    server_parser = modes.add_parser(
        "server", help="Serve incoming generation requests."
    )

    for command_parser, config_cls in (
        (train_parser, ExperimentConfig),
        (mapper_parser, InferenceMapperConfig),
        (server_parser, InferenceServerConfig),
    ):
        command_parser.set_defaults(config_cls=config_cls)
        command_parser.add_argument("action", choices=("run", "config"))
        command_parser.add_argument(
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

    match (args.command, args.mode):
        case ("train", None):
            from .training.run import run_training

            run_training(config)
        case ("inference", "mapper"):
            from .generation import run_offline_inference

            run_offline_inference(config)
        case ("inference", "server"):
            from .generation.server import run_inference_server

            run_inference_server(config)


if __name__ == "__main__":
    main()
