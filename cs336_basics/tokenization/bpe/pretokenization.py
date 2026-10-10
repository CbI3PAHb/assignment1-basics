"""Parallel pre-token counting for BPE training."""

from __future__ import annotations

import logging
import multiprocessing
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import regex as re
import tqdm

from cs336_basics.pretokenization_example import PAT

Token = bytes
Pretoken = tuple[Token, ...]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkTask:
    """
    Picklable input for one pre-tokenization worker.

    Half interval for [start_byte, end_byte)
    """

    input_path: Path
    start_byte: int
    end_byte: int
    special_tokens: tuple[str, ...]


def calculate_num_chunks(file_size: int, chunk_size: int) -> int:
    """Return the desired number of chunks, including for an empty file."""
    if file_size < 0:
        raise ValueError(f"file_size must be >= 0, got {file_size}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    return max(1, (file_size + chunk_size - 1) // chunk_size)


def find_safe_chunk_boundaries(
    input_path: Path,
    *,
    desired_num_chunks: int,
    split_token: bytes,
) -> list[int]:
    """Find boundaries that cannot split a pre-token.

    ``split_token`` must be a declared special token that is removed before
    pre-tokenization. If there is no safe delimiter, correctness is more
    important than parallelism and a single ``[0, file_size]`` chunk is valid.

    TODO:
        Implement boundary discovery without hardcoding ``<|endoftext|>``.
    """
    if desired_num_chunks <= 0:
        raise ValueError(f"desired_num_chunks must be > 0, got {desired_num_chunks}")

    if len(split_token) == 0:
        raise ValueError("split_token must not be empty")

    file_size = input_path.stat().st_size
    if file_size == 0:
        return [0, 0]

    if not isinstance(split_token, bytes):
        raise TypeError("Must represent special token as a bytestring")

    if desired_num_chunks == 1:
        return [0, file_size]

    approximate_chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * approximate_chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    with open(input_path, "rb") as file:
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            while True:
                file.seek(initial_position)  # Start at boundary guess
                mini_chunk = file.read(mini_chunk_size + len(split_token) - 1)  # Read a mini chunk + overlapping

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _count_pretokens_in_chunk(task: ChunkTask) -> Counter[Pretoken]:
    """Count pre-tokens in one byte range.

    This function runs in worker processes, so keep it top-level, deterministic,
    picklable, and independent of global log handlers.
    """
    with open(task.input_path, "rb") as file:
        file.seek(task.start_byte)
        chunk = file.read(task.end_byte - task.start_byte).decode("utf-8")

    special_token_pattern = "|".join(re.escape(token) for token in sorted(task.special_tokens, key=len, reverse=True))

    text_parts = re.split(special_token_pattern, chunk)
    freqs: Counter[Pretoken] = Counter()

    for text_part in text_parts:
        for match in re.finditer(PAT, text_part):
            pretoken: Pretoken = tuple(bytes([byte]) for byte in match.group().encode("utf-8"))
            freqs[pretoken] += 1

    return freqs


def _iter_chunk_counts(
    tasks: list[ChunkTask],
    num_workers: int,
) -> Iterator[Counter[Pretoken]]:
    """Yield counts from every chunk, serially or through a process pool."""
    if num_workers == 1:
        yield from map(_count_pretokens_in_chunk, tasks)
    else:
        with multiprocessing.Pool(num_workers) as pool:
            yield from pool.imap_unordered(_count_pretokens_in_chunk, tasks)


def count_pretokens(
    input_path: str | Path,
    *,
    special_tokens: tuple[str, ...],
    chunk_size: int,
    n_process: int,
) -> Counter[Pretoken]:
    """Count pre-tokens across an input corpus."""

    if n_process <= 0:
        raise ValueError(f"n_process must be > 0, got {n_process}")

    input_path = Path(input_path)
    file_size = input_path.stat().st_size

    desired_num_chunks = calculate_num_chunks(file_size=file_size, chunk_size=chunk_size)

    logger.info(
        "Counting pretokens: path=%s, size=%d, workers=%d, desired_chunks=%d",
        input_path,
        file_size,
        n_process,
        desired_num_chunks,
    )

    tasks = []
    chunk_boundaries = find_safe_chunk_boundaries(
        input_path=input_path,
        desired_num_chunks=desired_num_chunks,
        split_token=special_tokens[0].encode("utf-8"),
    )
    for start_byte, end_byte in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        task = ChunkTask(
            input_path=input_path,
            start_byte=start_byte,
            end_byte=end_byte,
            special_tokens=special_tokens,
        )
        tasks.append(task)
    num_workers = min(n_process, len(tasks))
    freqs: Counter[Pretoken] = Counter()

    for partial_count in tqdm.tqdm(_iter_chunk_counts(tasks, num_workers), total=len(tasks)):
        freqs.update(partial_count)

    logger.info(
        "Pretoken counting complete: unique=%d, total=%d",
        len(freqs),
        freqs.total(),
    )
    return freqs
