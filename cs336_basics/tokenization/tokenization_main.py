"""
owt_valid

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/tokenization_main.py \
    --dataset-file-path /Users/parii-artem/Documents/assignment1-basics/data/owt_valid.txt \
    --save-file-path /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/owt_valid_vocab.pkl \
    --json-log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/owt_valid_json_logs.json \
    --log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/owt_valid_app.log \
    --chunk-size 16777216 \
    --n-process 8

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/plot_bpe_stats.py \
    /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/owt_valid_json_logs.json


owt_train

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/tokenization_main.py \
    --dataset-file-path /Users/parii-artem/Documents/assignment1-basics/data/owt_train.txt \
    --save-file-path /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/owt_train_vocab.pkl \
    --json-log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/owt_train_json_logs.json \
    --log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/owt_train_app.log \
    --chunk-size 4194304 \
    --n-process 12

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/plot_bpe_stats.py \
    /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/owt_train_json_logs.json


tiny_stories_valid

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/tokenization_main.py \
    --dataset-file-path /Users/parii-artem/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt \
    --save-file-path /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/tiny_stories_valid_vocab.pkl \
    --json-log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/tiny_stories_valid_json_logs.json \
    --log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/tiny_stories_valid_app.log \
    --n-process 8

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/plot_bpe_stats.py \
    /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/tiny_stories_valid_json_logs.json


tiny_stories_train

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/tokenization_main.py \
    --dataset-file-path /Users/parii-artem/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt \
    --save-file-path /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/tiny_stories_train_vocab.pkl \
    --json-log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/tiny_stories_train_json_logs.json \
    --log-file /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/tiny_stories_train_app.log \
    --n-process 12

    --chunk-size 16777216 \
python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/plot_bpe_stats.py \
    /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/tiny_stories_train_json_logs.json
"""

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

from cs336_basics.pretokenization_example import PAT, find_chunk_boundaries


class ReverseLexOrderPair:
    """
    Encapsulates (bytes, bytes) so that in a min-heap, the "largest in normal lex order"
    is treated as the smallest. Ensures that tie frequencies pop in reverse lex order.
    """

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        # Invert normal order: self < other if self is > other (so larger lex sorts first).
        return self.pair > other.pair

    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair


logger = logging.getLogger(__name__)
json_logger = logging.getLogger("bpe_json_logger")


def setup_logging(args):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )

    # console logger
    if not args.silent:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)
        root_logger.info("Console logging enabled.")

    # file logger
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Text file logging enabled. Log file: {args.log_file}")

    # json logger
    if args.json_log_file:
        log_dir = os.path.dirname(args.json_log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        json_logger = logging.getLogger("bpe_json_logger")
        json_logger.setLevel(logging.INFO)
        json_logger.propagate = False  # Important

        if json_logger.hasHandlers():
            json_logger.handlers.clear()

        json_file_handler = logging.FileHandler(
            args.json_log_file, mode="a", encoding="utf-8"
        )
        json_formatter = logging.Formatter("%(message)s")
        json_file_handler.setFormatter(json_formatter)
        json_logger.addHandler(json_file_handler)
        root_logger.info(f"JSON logging enabled. Log file: {args.json_log_file}")

    if not root_logger.hasHandlers():
        root_logger.addHandler(logging.NullHandler())
        print("Warning: All logging is disabled (console, file, and JSON).")


def parse_args():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer.")
    parser.add_argument(
        "--dataset-file-path",
        type=str,
        help="Dataset file path (big .txt file).",
    )
    parser.add_argument(
        "--save-file-path",
        type=str,
        default="./bpe_tokenizer",
        help="Directory to save vocab and merges.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
    )
    parser.add_argument(
        "--n-process",
        type=int,
        default=os.cpu_count(),
        help="Num cpu process for pretokenization.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1 * 1024 * 1024,  # 1 MiB
        help="Chunk size in bytes (to prevent RAM OOM).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="if present - logging debug information in this file.",
    )
    parser.add_argument(
        "--json-log-file",
        type=str,
        default=False,
        help="path to save JSON-logs (for parsing and create graphs).",
    )
    parser.add_argument(
        "--silent", action="store_true", help="Disable output logs into console."
    )
    return parser.parse_args()


def calculate_num_chunks(file_path, desired_chunk_size: int = None):
    logger.debug(f"Calculating num chunks for file: {file_path}")

    try:
        with open(file_path, "rb") as file:
            # Get total file size in bytes
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            desired_chunk_size = (
                desired_chunk_size if desired_chunk_size is not None else 2**20
            )  #  1 MiB = 1024 KiB = 2 ** 20 B
            num_chunks = (file_size + desired_chunk_size - 1) // desired_chunk_size
            logger.info(
                f"Dataset file size: {file_size:,}, got {num_chunks:,} num chunks with {desired_chunk_size:,} desired chunk size"
            )
        return num_chunks
    except FileNotFoundError as e:
        logger.error(f"Cannot calculate chunks: input file not found at '{file_path}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error for file path '{file_path}'", exc_info=True)
        raise e


def _pretokenize(args):

    input_path, start_byte, end_byte, special_tokens, logger = args
    if not special_tokens:
        raise RuntimeError(f"There is no special tokens: {special_tokens}")

    logger.debug(
        f"{multiprocessing.current_process().name} - {start_byte=}, {end_byte=}: reading chunk"
    )

    byte_string_frequencies: dict[tuple[bytes], int] = dict()

    with open(input_path, "rb") as file:
        file.seek(start_byte)
        chunk = file.read(end_byte - start_byte).decode("utf-8", errors="ignore")

    # Run pre-tokenization on your chunk and store the counts for each pre-token
    # 3. Pre-tokenization
    split_pattern = "|".join(map(re.escape, special_tokens))
    # logger.debug(f"{special_tokens=}")

    sub_chunks = re.split(split_pattern, chunk)

    logger.debug(
        f"{multiprocessing.current_process().name} - {start_byte=}, {end_byte=}: pretokenization process..."
    )
    # logger.debug(sub_chunks)
    # logger.debug(f"{split_pattern=}")

    for sub_chunk in sub_chunks:
        if not sub_chunk or (sub_chunk in special_tokens):
            continue
        for pre_token in re.finditer(PAT, sub_chunk):
            byte_representation = tuple(
                bytes([b]) for b in pre_token.group().encode("utf-8")
            )
            byte_string_frequencies[byte_representation] = (
                byte_string_frequencies.get(byte_representation, 0) + 1
            )
    logger.info(
        f"{multiprocessing.current_process().name} - {start_byte=}, {end_byte=}: pretokenization process done"
    )
    return byte_string_frequencies


def create_new_byte_string(
    byte_string: tuple[bytes, ...],
    most_frequent_pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    byte_string_parts = []
    i = 0
    while i < len(byte_string):
        if (
            i + 1 < len(byte_string)
            and byte_string[i] == most_frequent_pair[0]
            and byte_string[i + 1] == most_frequent_pair[1]
        ):
            byte_string_parts.append(byte_string[i] + byte_string[i + 1])
            i += 2
        else:
            byte_string_parts.append(byte_string[i])
            i += 1
    new_byte_string: tuple[bytes, ...] = tuple(byte_string_parts)
    return new_byte_string


def bpeTrainingFunction(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    chunk_size: int = 1 * 2**20,
    n_process: int = 4,
    n_iters_to_brutforce_calculate_most_frequence_pair: int = 300,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    1. Vocabulary initialization
    2. Removing special tokens before pre-tokenization
    3. Pre-tokenization
    4. Compute BPE merges
    5. Special tokens
    Note: Optimizing the merging step with this caching procedure
    Note: Parallelizing pre-tokenization
    return: tuple(vocab, merges)
    """
    start_function_time = time.time()
    # 1. Vocabulary initialization
    # logger.info("Vocabulary initialization...")
    vocabulary: dict[int, bytes] = {b: bytes([b]) for b in range(256)}
    initial_vocab_size = len(vocabulary)

    # 2. Removing special tokens before pre-tokenization
    n_chunks = calculate_num_chunks(input_path, chunk_size)

    with open(input_path, "rb") as file:
        chunk_boundaries = find_chunk_boundaries(file, n_chunks, b"<|endoftext|>")
        # logger.debug(f"Chunk boundaries: {chunk_boundaries}")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
    byte_string_frequencies: dict[tuple[bytes], int] = dict()

    logger.info(f"start pre-tokenization process with {n_process} proc...")

    with multiprocessing.Pool(n_process) as pool:
        TASKS = []
        for i, (start_byte, end_byte) in enumerate(
            zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        ):
            args = (input_path, start_byte, end_byte, special_tokens, logger)
            TASKS.append(args)

        imap_unordered_it = pool.imap_unordered(_pretokenize, TASKS)

        for x in tqdm.tqdm(imap_unordered_it, total=len(TASKS)):
            for k, v in x.items():
                byte_string_frequencies[k] = byte_string_frequencies.get(k, 0) + v

    logger.info(f"Done pretokenization")
    end_pretokenization_process_time = time.time()

    # 4. Compute BPE merges
    logger.info("Start training tokenizer")

    merges: list[tuple[bytes, bytes]] = list()
    pair_frequencies: dict[tuple[bytes, bytes], int] = dict()
    pair_of_tokens_to_index: dict[tuple[bytes, bytes], set[int]] = dict()
    word_index_to_word: dict[int, tuple[bytes]] = dict()
    word_index_to_frequency: dict[int, int] = dict()

    for byte_string, frequency in byte_string_frequencies.items():
        for pair in zip(byte_string, byte_string[1:]):
            pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency

    for index, (byte_string, frequency) in enumerate(byte_string_frequencies.items()):
        word_index_to_word[index] = byte_string
        word_index_to_frequency[index] = frequency
        for pair in zip(byte_string, byte_string[1:]):
            if pair not in pair_of_tokens_to_index:
                pair_of_tokens_to_index[pair] = set()
            pair_of_tokens_to_index[pair].add(index)

    HEADER = f"{'Iter':>6} | {'Pairs':>10} | {'Indexes':>10} | {'Words':>8} | {'MergeIn':>10} | {'MeanLen':>8} | {'Time(ms)':>9}"
    logger.debug(HEADER)

    # for new_token_index in range(initial_vocab_size, vocab_size - len(special_tokens)):
    for new_token_index in tqdm.tqdm(
        range(initial_vocab_size, vocab_size - len(special_tokens))
    ):
        start_iteration = time.time()
        if not pair_frequencies:
            break

        # logger.debug(f"Adding new token with index {new_token_index:5d}")
        # logger.info("Counting initial pair frequencies")

        # logger.debug(f"{len(pair_frequencies)=}")
        # logger.debug(f"{len(pair_of_tokens_to_index)=}")
        # logger.debug(f"{len(word_index_to_word)=}")
        # logger.debug(f"{len(word_index_to_frequency)=}")

        if new_token_index < n_iters_to_brutforce_calculate_most_frequence_pair:
            most_frequent_pair = max(
                pair_frequencies, key=lambda x: (pair_frequencies.get(x, 0), x)
            )
        else:
            if new_token_index == n_iters_to_brutforce_calculate_most_frequence_pair:
                pq = [
                    (-freq, ReverseLexOrderPair(pair))
                    for pair, freq in pair_frequencies.items()
                ]
                heapq.heapify(pq)
            while pq:
                # potential most frequency pair - O(log N)
                neg_freq, pair = heapq.heappop(pq)
                pair = pair.pair

                # Important: compare potential pair frequency with true most frequency pair
                if -neg_freq == pair_frequencies.get(pair):
                    most_frequent_pair = pair
                    break

        new_byte_token = most_frequent_pair[0] + most_frequent_pair[1]

        # logger.debug(f"Most frequent pair is {most_frequent_pair} with freq = {pair_frequencies[most_frequent_pair]}")
        # logger.debug(f"Token to insert: {new_byte_token}")
        merges.append(most_frequent_pair)
        vocabulary[new_token_index] = new_byte_token

        word_indexes_where_new_byte_pair_exists: list[int] = list(
            pair_of_tokens_to_index[most_frequent_pair]
        )
        mean_current_byte_string_len = 0

        for word_index in word_indexes_where_new_byte_pair_exists:
            current_byte_string: tuple[bytes] = word_index_to_word[word_index]
            mean_current_byte_string_len += len(current_byte_string)
            frequency: int = word_index_to_frequency[word_index]

            # forming new word
            new_byte_string = create_new_byte_string(
                current_byte_string, most_frequent_pair
            )

            for pair in set(zip(current_byte_string, current_byte_string[1:])):
                pair_of_tokens_to_index[pair].remove(word_index)
                if not pair_of_tokens_to_index[pair]:
                    del pair_of_tokens_to_index[pair]

            for pair in zip(current_byte_string, current_byte_string[1:]):
                pair_frequencies[pair] -= frequency
                if pair_frequencies[pair] == 0:
                    del pair_frequencies[pair]
                elif (
                    new_token_index > n_iters_to_brutforce_calculate_most_frequence_pair
                ):
                    heapq.heappush(
                        pq, (-pair_frequencies[pair], ReverseLexOrderPair(pair))
                    )

            for pair in zip(new_byte_string, new_byte_string[1:]):
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency
                if pair not in pair_of_tokens_to_index:
                    pair_of_tokens_to_index[pair] = set()
                pair_of_tokens_to_index[pair].add(word_index)
                if new_token_index > n_iters_to_brutforce_calculate_most_frequence_pair:
                    heapq.heappush(
                        pq, (-pair_frequencies[pair], ReverseLexOrderPair(pair))
                    )

            word_index_to_word[word_index] = new_byte_string

        mean_current_byte_string_len /= (
            len(word_indexes_where_new_byte_pair_exists) or 1
        )

        log_data = {
            "iteration": new_token_index,
            "pair_frequencies_count": len(pair_frequencies),
            "pair_indexes_count": len(pair_of_tokens_to_index),
            "words_to_merge_count": len(word_indexes_where_new_byte_pair_exists),
            "time_ms": round((time.time() - start_iteration) * 1000, 2),
            "words_in_vocab_count": len(word_index_to_word),
            "mean_string_len": mean_current_byte_string_len,
        }

        json_logger.info(json.dumps(log_data))
        time_taken = round((time.time() - start_iteration) * 1000, 3)

        log_line = (
            f"{new_token_index:>6} | "
            f"{len(pair_frequencies):>10,} | "
            f"{len(pair_of_tokens_to_index):>10,} | "
            f"{len(word_index_to_word):>8,} | "
            f"{len(word_indexes_where_new_byte_pair_exists):>10,} | "
            f"{mean_current_byte_string_len:>8.2f} | "
            f"{time_taken:>9.2f}"
        )
        logger.debug(log_line)

    for special_token in special_tokens:
        vocabulary[len(vocabulary)] = special_token.encode("utf-8")

    end_training_bpe_time = time.time()
    logger.info(
        f"Pretokenization taken {end_pretokenization_process_time - start_function_time:.3f} sec"
    )
    logger.info(
        f"Train BPE tokenizer taken {end_training_bpe_time - end_pretokenization_process_time:.3f} sec"
    )
    logger.info(f"Total time taken: {end_training_bpe_time - start_function_time:.3f}")
    return vocabulary, merges


def main():
    args = parse_args()

    setup_logging(args)

    # # logger = logging.getLogger(__name__)
    # # json_logger = logging.getLogger("bpe_json_logger")
    # logger.info("Starting process...")
    # logger.debug(f"Recieved arguments: {args}")

    # vocab, merges = bpeTrainingFunction(
    #     input_path=args.dataset_file_path,
    #     vocab_size=args.vocab_size,
    #     special_tokens=["<|endoftext|>"],
    #     chunk_size=args.chunk_size,
    #     n_process=args.n_process,
    #     n_iters_to_brutforce_calculate_most_frequence_pair=3000,
    #     # logger
    # )

    # with open(args.save_file_path, "wb") as f:
    #     pickle.dump({"vocab": vocab, "merges": merges}, f)

    # with open(args.save_file_path, "rb") as f:
    #     loaded_data = pickle.load(f)

    # print(f"{type(loaded_data)=}")
    # print(f"{loaded_data.keys()=}")

    # vocab = loaded_data['vocab']
    # merges = loaded_data['merges']

    # print(f"type(vocab): {type(vocab)}")
    # print(f"type(merges): {type(merges)}")
    # print(f"vocab len: {len(vocab)}")
    # print(f"num merges: {len(merges)}")


if __name__ == "__main__":
    main()
