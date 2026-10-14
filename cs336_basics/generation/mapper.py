from cs336_basics.training.config.inference import InferenceMapperConfig


def read_prompt(config: InferenceMapperConfig) -> str:
    """Read either literal prompt text or the contents of a UTF-8 input file"""

    raise NotImplementedError("Mapper input reading is not implemented yet")


def run_inference_mapper(config: InferenceMapperConfig) -> None:
    """Load one engine, read the input, generate and save output, then exit"""

    raise NotImplementedError("Mapper inference is not implemented yet")
