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


def get_most_frequence_pair(pair_frequence):
    sorted_pairs: list[tuple[tuple[bytes, bytes], int]] = sorted([item for item in pair_frequence.items()], key=lambda x: (x[1], x[0]), reverse=True)
    return sorted_pairs[0][0]


def calculate_num_chunks(file_path, desired_chunk_size: int = None):
    with open(file_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        desired_chunk_size = (
            desired_chunk_size if desired_chunk_size is not None else 2**20
        )  #  1 MiB = 1024 KiB = 2 ** 20 B
        num_chunks = (file_size + desired_chunk_size - 1) // desired_chunk_size
    return num_chunks


def _pretokenize(args):

    input_path, start_byte, end_byte, special_tokens = args
    if not special_tokens:
        raise RuntimeError(f"There is no special tokens: {special_tokens}")

    byte_string_frequencies: dict[tuple[bytes], int] = dict()

    with open(input_path, "rb") as file:
        file.seek(start_byte)
        chunk = file.read(end_byte - start_byte).decode("utf-8", errors="ignore")

    # Run pre-tokenization on your chunk and store the counts for each pre-token
    # 3. Pre-tokenization
    split_pattern = "|".join(map(re.escape, special_tokens))

    sub_chunks = re.split(split_pattern, chunk)

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
    return byte_string_frequencies

def merge(
    byte_string_frequencies: dict[tuple[bytes, ...], int], 
    most_frequency_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    new_byte_string_frequencies:  dict[tuple[bytes, ...], int] = dict()
    for byte_string, byte_string_frequency in byte_string_frequencies.items():
        byte_string_parts = []
        i = 0
        while i < len(byte_string):
            if i + 1 < len(byte_string) and byte_string[i] == most_frequency_pair[0] and byte_string[i + 1] == most_frequency_pair[1]:
                byte_string_parts.append(byte_string[i] + byte_string[i + 1])
                i += 2
            else:
                byte_string_parts.append(byte_string[i])
                i += 1
        new_byte_string: tuple[bytes, ...] = tuple(byte_string_parts)
        new_byte_string_frequencies[new_byte_string] = byte_string_frequency
    return new_byte_string_frequencies


def update_frequencies(
    byte_string_frequencies: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    pair_frequencies          : dict[tuple[bytes, bytes], int]      = dict()
    for byte_string, frequency in byte_string_frequencies.items():
        for pair in zip(byte_string, byte_string[1:]):
            pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency
    return pair_frequencies


def bpeTrainingFunction(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    chunk_size: int = 1 * 2**20,
    n_process: int = 4,
    n_iters_to_brutforce_calculate_most_frequence_pair: int = 3000,
):
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
    # 1. Vocabulary initialization
    # logger.info("Vocabulary initialization...")
    init_vocab_size = 256
    vocab: dict[int, bytes] = {b: bytes([b]) for b in range(256)}
    initial_vocab_size = len(vocab)

    # 2. Removing special tokens before pre-tokenization
    n_chunks = calculate_num_chunks(input_path, chunk_size)

    with open(input_path, "rb") as file:
        chunk_boundaries = find_chunk_boundaries(file, n_chunks, b"<|endoftext|>")
        # logger.debug(f"Chunk boundaries: {chunk_boundaries}")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
    byte_string_frequencies: dict[tuple[bytes], int] = dict()

    with multiprocessing.Pool(n_process) as pool:
        TASKS = []
        for i, (start_byte, end_byte) in enumerate(
            zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        ):
            args = (input_path, start_byte, end_byte, special_tokens)
            TASKS.append(args)

        imap_unordered_it = pool.imap_unordered(_pretokenize, TASKS)

        for x in tqdm.tqdm(imap_unordered_it, total=len(TASKS)):
            for k, v in x.items():
                byte_string_frequencies[k] = byte_string_frequencies.get(k, 0) + v

    end_pretokenization_process_time = time.time()

    merges                    : list[tuple[bytes, bytes]]           = list()
    pair_frequencies          : dict[tuple[bytes, bytes], int]      = dict()

    for byte_string, frequency in byte_string_frequencies.items():
        for pair in zip(byte_string, byte_string[1:]):
            pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency

    for new_token_index in range(init_vocab_size, vocab_size - len(special_tokens)):
        if not pair_frequencies:
            break
        most_frequency_pair: tuple[bytes, bytes] = get_most_frequence_pair(pair_frequencies)
        merges.append(most_frequency_pair)
        t1, t2 = most_frequency_pair
        new_token = t1 + t2
        vocab[new_token_index] = new_token

        # смержить t1 + t2 во всех словах
        byte_string_frequencies = merge(byte_string_frequencies, most_frequency_pair)
        # обновить частоты пар
        pair_frequencies = update_frequencies(byte_string_frequencies)
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab, merges