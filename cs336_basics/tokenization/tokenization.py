import os
import logging
import multiprocessing
import regex as re

from cs336_basics.pretokenization_example import PAT, find_chunk_boundaries


logger = logging.getLogger(__name__) 
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Показывать на консоли только INFO и выше

file_handler = logging.FileHandler('cs336_basics/tokenization/app.log', mode='w')
file_handler.setLevel(logging.DEBUG) # Записывать в файл всё, включая DEBUG

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Preparing main....")


# def pretokenize(chunk: str, byte_string_frequencies: dict[tuple[bytes], int], special_tokens: list[str]):
#     split_pattern = f"({'|'.join(map(re.escape, special_tokens))})"
#     sub_chunks = re.split(split_pattern, chunk)

#     for sub_chunk in sub_chunks:
#         if not sub_chunk or (sub_chunk in special_tokens):
#             continue
#         for pre_token in re.finditer(PAT, sub_chunk):
#             byte_representation = tuple(bytes([b]) for b in pre_token.group(0).encode("utf-8"))
#             byte_string_frequencies[byte_representation] = byte_string_frequencies.get(byte_representation, 0) + 1


def calculate_num_chunks(file_path, desired_chunk_size: int = None):
    with open(file_path, "rb") as file:
    # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        desired_chunk_size = desired_chunk_size if desired_chunk_size is not None else 2 * 2 ** 20 #  1 MiB = 1024 KiB = 2 ** 20 B
        num_chunks = (file_size + desired_chunk_size - 1) // desired_chunk_size
        logger.info(f"{file_size=}, {num_chunks=}, {desired_chunk_size=}")
    return num_chunks


def _pretokenize(args):
    input_path, start_byte, end_byte, special_tokens = args
    if not special_tokens:
        raise RuntimeError(f"There is no special tokens: {special_tokens}")

    byte_string_frequencies: dict[tuple[bytes], int] = dict()

    logger.info(f"Reading chunk: {start_byte=}: {end_byte=}")

    with open(input_path, "rb") as file:    
        file.seek(start_byte)
        chunk = file.read(end_byte - start_byte).decode("utf-8", errors="ignore")

    logger.info(f"Reading chunk: {start_byte=}: {end_byte=} done")

    # Run pre-tokenization on your chunk and store the counts for each pre-token
    # 3. Pre-tokenization
    # logger.info(f"Pretokenization #{i:04d} chunk")
    split_pattern = '|'.join(map(re.escape, special_tokens))
    # logger.debug(f"{special_tokens=}")

    sub_chunks = re.split(split_pattern, chunk)

    logger.info(f"Pretokenization process...")
    # logger.debug(sub_chunks)
    # logger.debug(f"{split_pattern=}")

    for sub_chunk in sub_chunks:
        if not sub_chunk or (sub_chunk in special_tokens):
            continue
        for pre_token in re.finditer(PAT, sub_chunk):
            byte_representation = tuple(bytes([b]) for b in pre_token.group().encode("utf-8"))
            byte_string_frequencies[byte_representation] = byte_string_frequencies.get(byte_representation, 0) + 1
    return byte_string_frequencies


def bpeTrainingFunction(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
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
    # 1. Vocabulary initialization
    # logger.info("Vocabulary initialization...")
    vocabulary: dict[int, bytes] = {b: bytes([b]) for b in range(256)}
    initial_vocab_size = len(vocabulary)

    # 2. Removing special tokens before pre-tokenization
    logger.info(f"Calculate num chunks")
    n_chunks = calculate_num_chunks(input_path)

    logger.info("Removing special tokens before pre-tokenization")

    with open(input_path, "rb") as file:    
        chunk_boundaries = find_chunk_boundaries(file, n_chunks, b"<|endoftext|>")
        # logger.debug(f"Chunk boundaries: {chunk_boundaries}")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
    byte_string_frequencies: dict[tuple[bytes], int] = dict()

    logger.info("Pre-tokenization process...")

    for i, (start_byte, end_byte) in enumerate(zip(chunk_boundaries[:-1], chunk_boundaries[1:])):
        logger.info(f"Pretokenizing #{i:4d} chunk")
        args = (input_path, start_byte, end_byte, special_tokens)
        pretokenized_byte_string_frequencies = _pretokenize(args)

        logger.info(f"Pretokenizing #{i:4d} done, update byte_string_frequencies")
        for byte_string, frequency in pretokenized_byte_string_frequencies.items():
            byte_string_frequencies[byte_string] = byte_string_frequencies.get(byte_string, 0) + frequency

    logger.info(f"Done pretokenization")
    logger.debug(f"Byte string frequencies: {byte_string_frequencies}")

    # 4. Compute BPE merges
    logger.info("Start training tokenizer")
    merges: list[tuple[bytes, bytes]] = []

    for new_token_index in range(initial_vocab_size, vocab_size - len(special_tokens)):
        # logger.info(f"Adding new token with index {new_token_index:5d}")
        # logger.info("Counting initial pair frequencies")

        pair_frequencies: dict[tuple[bytes], int] = {}
        for byte_string, frequency in byte_string_frequencies.items():
            for pair in zip(byte_string, byte_string[1:]):
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + frequency
        # logger.debug(f"Pair frequencies: {pair_frequencies}")

        most_frequent_pair = max(pair_frequencies, key=lambda x: (pair_frequencies.get(x, 0), x))
        new_byte_token = most_frequent_pair[0] + most_frequent_pair[1]
        
        # logger.debug(f"Most frequent pair is {most_frequent_pair} with freq = {pair_frequencies[most_frequent_pair]}")
        # logger.debug(f"Token to insert: {new_byte_token}")
        merges.append(most_frequent_pair)
        vocabulary[new_token_index] = new_byte_token

        new_byte_string_frequencies: dict[tuple[bytes], int] = dict()

        for byte_string, frequency in byte_string_frequencies.items():
            i = 0
            string_parts = []
            while i < len(byte_string):
                if i + 1 < len(byte_string) and (byte_string[i] + byte_string[i + 1]) == new_byte_token:
                    string_parts.append(new_byte_token)
                    i += 2
                else:
                    string_parts.append(byte_string[i])
                    i += 1
            new_byte_string = tuple(string_parts)
            new_byte_string_frequencies[new_byte_string] = new_byte_string_frequencies.get(new_byte_string, 0) + frequency
        byte_string_frequencies = new_byte_string_frequencies
        # logger.debug(f"new_byte_string_frequencies: {new_byte_string_frequencies}")
        
    for special_token in special_tokens:
        vocabulary[len(vocabulary)] = special_token.encode('utf-8')
    return vocabulary, merges



if __name__ == "__main__":
    vocab, merges = bpeTrainingFunction("/Users/parii-artem/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 512, ['<|endoftext|>'])
    # bpeTrainingFunction("/Users/parii-artem/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 256, [])
