import argparse
import heapq
import json
import logging
import multiprocessing
import os
import pickle
import time

import regex as re
import tqdm

from cs336_basics.logging_setup import setup_logging
from cs336_basics.tokenization.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer.")
    parser.add_argument(
        "--dataset-file-path",
        type=str,
        help="Dataset file path (big .txt file).",
    )
    parser.add_argument(
        "--vocab-file-path",
        type=str,
        default="./bpe_tokenizer",
        help="Directory to load vocab and merges.",
    )
    parser.add_argument(
        "--silent", action="store_true", help="Disable output logs into console."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(silent=args.silent)

    logger = logging.getLogger(__name__)
    logger.info("Starting process...")
    logger.debug(f"Recieved arguments: {args}")
    logger.debug(f"{args.dataset_file_path=}")
    logger.debug(f"{args.vocab_file_path=}")

    with open(args.vocab_file_path, "rb") as f:
        loaded_data = pickle.load(f)

    logger.info(f"{type(loaded_data)=}")
    logger.info(f"{loaded_data.keys()=}")

    vocab = loaded_data["vocab"]
    merges = loaded_data["merges"]

    logger.info(f"type(vocab): {type(vocab)}")
    logger.info(f"type(merges): {type(merges)}")
    logger.info(f"vocab len: {len(vocab)}")
    logger.info(f"num merges: {len(merges)}")

    tokneizer: Tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    start_time = time.time()
    with open(args.dataset_file_path, "r") as f:
        data = f.read()
    end_reading_file = time.time()
    logger.info(f"time taken to read data: {(end_reading_file - start_time):.3f} sec")

    indexes = tokneizer.encode(data)
    logger.info(f"len(indexes) = {len(indexes)}")
    logger.info(f"time taken to encode: {(time.time() - end_reading_file):.3f} sec")


if __name__ == "__main__":
    main()

"""
python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/sandbox/tokenize_text.py \
    --dataset-file-path /Users/parii-artem/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt \
    --vocab-file-path /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/owt_train_vocab.pkl \
    > owt_train.stdout.log 2> owt_train.stderr.log


python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/sandbox/tokenize_text.py \
    --dataset-file-path /Users/parii-artem/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt \
    --vocab-file-path /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/tiny_stories_train_vocab.pkl \
    > tiny_stories.stdout.log 2> tiny_stories.stderr.log
"""
