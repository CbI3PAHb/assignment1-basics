import logging

import regex as re
from cs336_basics.pretokenization_example import PAT

# --- 1. Настройка логгера (исправлено getLogger) ---
# Использование name — это стандартная практика.
# Python автоматически присваивает этой переменной имя текущего модуля.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Показывать на консоли только INFO и выше

file_handler = logging.FileHandler("app.log", mode="w")
file_handler.setLevel(logging.DEBUG)  # Записывать в файл всё, включая DEBUG

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Preparing main....")


def main():
    logger.info("Script started.")

    strings = ["low"] * 5 + ["lower"] * 2 + ["widest"] * 3 + ["newest"] * 6
    string = " ".join(strings)

    pretokenized_strings = string.split(" ")

    logger.debug("Pre-tokenized strings: %s", pretokenized_strings)

    initial_frequencies: dict[tuple[bytes], int] = {}
    for s in pretokenized_strings:
        if not s:
            continue
        byte_string = tuple(bytes([b]) for b in s.encode("utf-8"))
        initial_frequencies[byte_string] = initial_frequencies.get(byte_string, 0) + 1

    merges = []

    for i in range(6):
        frequencies: dict[tuple[bytes, bytes], int] = {}

        logger.info("Counting frequencies....")
        for subtokenized_tuple, count in initial_frequencies.items():
            for b1, b2 in zip(subtokenized_tuple, subtokenized_tuple[1:]):
                pair = (b1, b2)
                frequencies[pair] = frequencies.get(pair, 0) + count

        logger.info("Getting most frequent pair in frequencies...")

        logger.debug(f"Given freqeuencies: {frequencies}")
        most_frequent_pair = max(frequencies, key=lambda x: (frequencies.get(x, 0), x))
        logger.info("Most frequent pair is %s", most_frequent_pair)
        pair_to_merge = most_frequent_pair[0] + most_frequent_pair[1]
        merges.append(pair_to_merge)
        logger.info(
            "Bytes pair %s, type(pair) = %s", pair_to_merge, type(pair_to_merge)
        )

        logger.info("Now we need to change every this pair of bytes on merged...")

        new_initial_frequencies: dict[tuple[bytes, bytes], int] = {}

        for world_tuple, count in initial_frequencies.items():
            logger.info(f"processing tuple: {world_tuple}, with count: {count}")
            new_world_parts = []
            i = 0
            while i < len(world_tuple):
                if (
                    i + 1 < len(world_tuple)
                    and (world_tuple[i] + world_tuple[i + 1]) == pair_to_merge
                ):
                    new_world_parts.append(pair_to_merge)
                    logger.debug(
                        f"in {world_tuple} founded pair {pair_to_merge} at position {(i, i+1)}"
                    )
                    i += 2
                else:
                    new_world_parts.append(world_tuple[i])
                    i += 1
            merged_word_tuple = tuple(new_world_parts)
            new_initial_frequencies[merged_word_tuple] = (
                new_initial_frequencies.get(merged_word_tuple, 0) + count
            )

        logger.debug(f"frequencies after merge: {new_initial_frequencies}")
        initial_frequencies = new_initial_frequencies
    logger.debug(f"merges: {merges}")
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # logger.exception() автоматически добавляет в лог полный traceback ошибки.
        logger.exception("An unhandled error occurred during script execution!")
