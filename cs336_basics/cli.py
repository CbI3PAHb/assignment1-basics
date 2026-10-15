"""Command-line entrypoint for training, mapper inference and serving"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config.experiment import ExperimentConfig
from .config.inference import InferenceMapperConfig, InferenceServerConfig
from .config.io import dump_config, load_config


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

    if args.action == "config":
        print(dump_config(config), end="")
        return

    match (args.command, args.mode):
        case ("train", None):
            from .training.run import run_training

            run_training(config)
        case ("inference", "mapper"):
            from .generation.mapper import run_inference_mapper

            run_inference_mapper(config)
        case ("inference", "server"):
            from .generation.server import run_inference_server

            run_inference_server(config)


if __name__ == "__main__":
    main()
