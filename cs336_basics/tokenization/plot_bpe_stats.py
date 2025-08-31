"""
python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/plot_bpe_stats.py \
    /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/json_logs.json

python3 /Users/parii-artem/Documents/assignment1-basics/cs336_basics/tokenization/plot_bpe_stats.py \
    /Users/parii-artem/Documents/assignment1-basics/cs336_basics/logs/owt_train_json_logs.json
"""

import json
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def plot_bpe_training_log(log_file="bpe_training_log.jsonl"):
    """
    Читает лог-файл в формате JSON Lines, обрабатывает его с помощью pandas
    и строит графики для анализа тренировки BPE.
    """
    try:
        # Читаем данные. Каждая строка - это JSON, который мы загружаем в список словарей.
        with open(log_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(
            f"Ошибка: Файл '{log_file}' не найден. Убедитесь, что вы запустили скрипт тренировки.",
            file=sys.stderr,
        )
        return
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON в файле '{log_file}': {e}", file=sys.stderr)
        return

    if not data:
        print(f"Файл '{log_file}' пуст. Нет данных для визуализации.", file=sys.stderr)
        return

    # Преобразуем список словарей в DataFrame - это очень удобно для анализа
    df = pd.DataFrame(data)
    df.set_index("iteration", inplace=True)
    df.sort_index(inplace=True)

    # --- Создаем окно для графиков ---
    # 2x2 сетка графиков, с общим размером окна
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Анализ тренировки BPE", fontsize=18)

    # Форматтер для оси Y, чтобы числа были читаемыми (например, 30,000 вместо 30000)
    comma_formatter = mticker.FuncFormatter(lambda x, p: format(int(x), ","))

    # --- 1. График: "Комбинаторный взрыв" размера словаря пар ---
    ax1 = axes[0, 0]
    df["pair_frequencies_count"].plot(ax=ax1, color="b", label="Кол-во уникальных пар")
    # ax1.set_yscale('log')
    ax1.set_title("Рост словаря пар (pair_frequencies)")
    ax1.set_ylabel("Количество пар")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.yaxis.set_major_formatter(comma_formatter)
    ax1.legend()

    # --- 2. График: Время выполнения итерации ---
    ax2 = axes[0, 1]
    df["time_ms"].plot(ax=ax2, color="r", label="Время на итерацию (мс)")
    # Добавим скользящее среднее для сглаживания пиков
    df["time_ms"].rolling(window=50).mean().plot(
        ax=ax2, color="darkred", linestyle="--", label="Скользящее среднее (50 итер.)"
    )
    ax2.set_yscale("log")
    ax2.set_title("Производительность: Время на итерацию")
    ax2.set_ylabel("Время (мс)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # --- 3. График: Количество слов, затронутых слиянием ---
    ax3 = axes[1, 0]
    df["words_to_merge_count"].plot(ax=ax3, color="g", label="Кол-во слов")
    df["words_to_merge_count"].rolling(window=50).mean().plot(
        ax=ax3, color="darkgreen", linestyle="--", label="Скользящее среднее (50 итер.)"
    )
    ax3.set_yscale("log")
    ax3.set_title("Область применения слияния")
    ax3.set_xlabel("Итерация (новый токен)")
    ax3.set_ylabel("Кол-во слов, где найдена пара")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.yaxis.set_major_formatter(comma_formatter)
    ax3.legend()

    # --- 4. График: Средняя длина изменяемой строки ---
    ax4 = axes[1, 1]
    df["mean_string_len"].plot(ax=ax4, color="purple")
    ax4.set_yscale("log")
    ax4.set_title("Средняя длина токенов в изменяемых словах")
    ax4.set_xlabel("Итерация (новый токен)")
    ax4.set_ylabel("Среднее кол-во токенов")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    # Автоматически подгоняем расположение элементов, чтобы они не перекрывались
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Сохраняем в файл и показываем на экране
    plt.savefig("bpe_training_analysis.png")
    plt.show()


if __name__ == "__main__":
    # Можно передать имя файла как аргумент командной строки
    log_file_path = sys.argv[1] if len(sys.argv) > 1 else "bpe_training_log_fast.jsonl"
    plot_bpe_training_log(log_file_path)
