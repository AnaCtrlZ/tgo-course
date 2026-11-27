"""Лабораторная работа 01: биграммная языковая модель и BPE-токенизатор с нуля.

Скрипт скачивает фиксированный корпус «Война и мир», строит биграммную модель
со сглаживанием Лапласа, оценивает её на отложенной части текста и обучает
BPE-токенизатор на всём корпусе. Используется только стандартная библиотека
Python.

Все пути считаются от папки этого файла, поэтому скрипт можно запускать из
любого рабочего каталога. Результаты записываются в ``pt01/artifacts/``:

* ``bigram.json`` -- статистика корпуса, частые слова, перплексии,
  правдоподобия и образцы генерации;
* ``bpe.json`` -- выученные слияния (сам токенизатор), размер словаря,
  статистика сжатия и примеры разбиения.

Запуск::

    python pt01/run.py            # полный прогон: 1500 слияний BPE, ~2 минуты
    python pt01/run.py --fast     # быстрый прогон: 200 слияний BPE, меньше минуты
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Глобальные константы
# ---------------------------------------------------------------------------

# Фиксированный корпус из задания.
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/"
    "word2vec-russian-novels/master/books_before/WarAndPeace.txt"
)

# Все пути считаются от каталога, где лежит этот скрипт.
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "corpus.txt"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Регулярное выражение для слов: только русские буквы, дефисные составные слова разрешены.
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")

# Маркер конца слова нужен BPE, чтобы различать, например, «он» и «она».
END_OF_WORD = "</w>"

# Фиксированные seed'ы делают каждый запуск воспроизводимым.
GENERATION_SEED = 12345
SHUFFLE_SEED = 777

# Параметры, зафиксированные заданием.
VAL_FRACTION = 0.10
LIKELIHOOD_ALPHA = 0.01
PERPLEXITY_ALPHAS = (1.0, 0.1, 0.01)
GENERATION_TEMPERATURES = (0.5, 1.0, 2.0)
GENERATION_STARTS = ("князь", "она")
GENERATION_LENGTH = 30

# Значения BPE по умолчанию: полный прогон — минимум 1500 слияний, --fast — 200.
FULL_MERGES = 1500
FAST_MERGES = 200
FULL_MIN_FREQUENCY = 2
FAST_MIN_FREQUENCY = 3


# ---------------------------------------------------------------------------
# Загрузка корпуса
# ---------------------------------------------------------------------------


def download_corpus(url: str, destination: Path) -> None:
    """Скачивает ``url`` в ``destination``, если кэша ещё нет."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[корпус] использую кэш {destination}")
        return

    print(f"[корпус] скачиваю {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "kilo-lab01"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    destination.write_bytes(data)
    print(f"[корпус] сохранено {len(data)} байт в {destination}")


def load_words(path: Path) -> list[str]:
    """Читает корпус, приводит к нижнему регистру, меняет «ё» на «е» и выделяет слова."""
    text = path.read_text(encoding="utf-8", errors="replace")
    normalized = text.lower().replace("ё", "е")
    return WORD_RE.findall(normalized)


def split_train_val(words: list[str], fraction: float) -> tuple[list[str], list[str]]:
    """Делит ``words`` на префикс для обучения и хвост для проверки.

    Разбиение идёт по словам (не по предложениям), как требует задание:
    последняя доля ``fraction`` всех словоупотреблений откладывается.
    """
    total = len(words)
    val_size = max(1, int(total * fraction))
    return words[: total - val_size], words[total - val_size :]


def most_common_pairs(counts: Counter, limit: int = 20) -> list[list]:
    """Возвращает ``limit`` самых частых слов в виде пар ``[слово, счётчик]``.

    При равных счётчиках порядок алфавитный, чтобы результат был детерминированным.
    """
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [[word, count] for word, count in ordered[:limit]]


# ---------------------------------------------------------------------------
# Биграммная языковая модель
# ---------------------------------------------------------------------------


class BigramModel:
    """Биграммная языковая модель, оцениваемая подсчётом пар соседних слов.

    Модель хранит сырые счётчики и превращает их в вероятности по запросу::

        P(w | v) = (c(v, w) + alpha) / (c(v) + alpha * V)

    где ``V`` — размер обучающего словаря. Аддитивное (лапласовское) сглаживание
    принципиально: без него каждая невиденная пара давала бы нулевую вероятность,
    а логарифм правдоподобия отложенного текста уходил бы в минус бесконечность.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Создаёт пустую модель с силой сглаживания ``alpha``."""
        self.alpha = float(alpha)
        self.word_counts: Counter = Counter()
        self.bigram_counts: dict[str, Counter] = {}
        self.vocab: list[str] = []
        self.vocab_size = 0

    def fit(self, words: list[str]) -> "BigramModel":
        """Оценивает частоты слов и биграмм по списку обучающих слов."""
        self.word_counts = Counter(words)
        self.vocab = sorted(self.word_counts)
        self.vocab_size = len(self.vocab)

        bigrams: dict[str, Counter] = {}
        for left, right in zip(words, words[1:]):
            row = bigrams.get(left)
            if row is None:
                row = Counter()
                bigrams[left] = row
            row[right] += 1
        self.bigram_counts = bigrams
        return self

    def count(self, left: str, right: str) -> int:
        """Возвращает обучающий счётчик биграммы ``(left, right)``."""
        return self.bigram_counts.get(left, {}).get(right, 0)

    def prob(self, v: str, w: str, alpha: float | None = None) -> float:
        """Возвращает сглаженную условную вероятность ``P(w | v)``.

        Если задан ``alpha``, он переопределяет значение модели по умолчанию;
        это позволяет оценивать одни и те же счётчики при разной силе сглаживания.
        """
        a = self.alpha if alpha is None else float(alpha)
        vocabulary = self.vocab_size if self.vocab_size > 0 else 1
        denominator = self.word_counts.get(v, 0) + a * vocabulary
        numerator = self.count(v, w) + a
        return numerator / denominator

    def logprob(self, words: list[str], alpha: float | None = None) -> float:
        """Возвращает ``sum_t log P(w_t | w_{t-1})``; первое слово не оценивается."""
        total = 0.0
        for previous, current in zip(words, words[1:]):
            total += math.log(self.prob(previous, current, alpha))
        return total

    def _next_distribution(self, context: str, temperature: float) -> list[float]:
        """Возвращает ненормированные ``P(w | context) ** (1 / T)`` по словарю."""
        a = self.alpha
        vocabulary = self.vocab_size if self.vocab_size > 0 else 1
        denominator = self.word_counts.get(context, 0) + a * vocabulary
        row = self.bigram_counts.get(context, {})
        power = 1.0 / temperature
        weights = []
        for word in self.vocab:
            probability = (row.get(word, 0) + a) / denominator
            weights.append(probability if power == 1.0 else probability**power)
        return weights

    def generate(
        self,
        start: str,
        length: int = GENERATION_LENGTH,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Сэмплирует текст длины ``length``, начиная со слова ``start``.

        Каждое следующее слово выбирается из строки предыдущего слова после
        возведения вероятностей в степень ``1 / temperature`` и перенормировки.
        Фиксированный экземпляр ГПСЧ делает выборку воспроизводимой.
        """
        generator = rng if rng is not None else random.Random(GENERATION_SEED)
        result = [start]
        while len(result) < length:
            context = result[-1]
            weights = self._next_distribution(context, temperature)
            total = sum(weights)
            threshold = generator.random() * total
            cumulative = 0.0
            chosen = self.vocab[-1]
            for word, weight in zip(self.vocab, weights):
                cumulative += weight
                if cumulative >= threshold:
                    chosen = word
                    break
            result.append(chosen)
        return result


def count_unseen_pairs(model: BigramModel, words: list[str]) -> int:
    """Считает пары в ``words``, которых не было в обучающих счётчиках."""
    unseen = 0
    for left, right in zip(words, words[1:]):
        if model.count(left, right) == 0:
            unseen += 1
    return unseen


def perplexity(model: BigramModel, words: list[str], alpha: float) -> float:
    """Возвращает биграммную перплексию ``words`` при сглаживании ``alpha``.

    У первого слова нет предыдущего контекста, поэтому оно не оценивается и
    ``N`` равно ``len(words) - 1``.
    """
    transitions = max(1, len(words) - 1)
    total_logprob = model.logprob(words, alpha)
    return math.exp(-total_logprob / transitions)


# ---------------------------------------------------------------------------
# BPE-токенизатор
# ---------------------------------------------------------------------------


class BPETokenizer:
    """BPE-токенизатор, обучаемый жадным слиянием самых частых пар.

    Слова рассматриваются как последовательности символов с добавленным маркером
    ``</w>``. Обучение многократно находит самую частую пару соседних символов по
    всему словарю, запоминает её как слияние и применяет ко всем словам.
    Упорядоченный список слияний *и есть* токенизатор.
    """

    def __init__(self, min_frequency: int = 2) -> None:
        """Создаёт необученный токенизатор, игнорирующий слова реже ``min_frequency``."""
        self.min_frequency = int(min_frequency)
        self.merges: list[list[str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self._cache: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
        """Заменяет все непересекающиеся вхождения ``pair`` их склейкой."""
        left, right = pair
        result: list[str] = []
        index = 0
        size = len(symbols)
        while index < size:
            if (
                index + 1 < size
                and symbols[index] == left
                and symbols[index + 1] == right
            ):
                result.append(left + right)
                index += 2
            else:
                result.append(symbols[index])
                index += 1
        return result

    def fit(self, word_freqs: dict[str, int], num_merges: int) -> "BPETokenizer":
        """Выучивает до ``num_merges`` слияний по отображению слово → частота.

        Счётчики пар и обратные индексы обновляются инкрементально, поэтому заново
        разбираются только слова, содержащие победившую пару. Это позволяет
        прогону на 1500 слияний уложиться в разумное время на чистом Python.
        """
        items = sorted(
            (word, freq)
            for word, freq in word_freqs.items()
            if freq >= self.min_frequency
        )

        splits: list[list[str]] = []
        frequencies: list[int] = []
        pair_counts: dict[tuple[str, str], int] = {}
        pair_words: dict[tuple[str, str], set[int]] = {}
        # buckets[c] хранит все пары с текущим счётчиком c; max_count — верхняя
        # граница, которая может только уменьшаться, поэтому лучшую пару можно
        # найти, не просматривая всю таблицу пар.
        buckets: dict[int, set[tuple[str, str]]] = {}

        for word_id, (word, frequency) in enumerate(items):
            symbols = list(word) + [END_OF_WORD]
            splits.append(symbols)
            frequencies.append(frequency)
            for pair in zip(symbols, symbols[1:]):
                previous = pair_counts.get(pair)
                if previous is None:
                    pair_counts[pair] = frequency
                    pair_words[pair] = {word_id}
                    buckets.setdefault(frequency, set()).add(pair)
                else:
                    pair_counts[pair] = previous + frequency
                    pair_words[pair].add(word_id)
                    buckets[previous].discard(pair)
                    buckets.setdefault(previous + frequency, set()).add(pair)

        max_count = max(pair_counts.values(), default=0)
        merges: list[list[str]] = []

        while len(merges) < num_merges:
            while max_count > 0 and not buckets.get(max_count):
                max_count -= 1
            if max_count <= 0:
                break

            best_pair = min(buckets[max_count])
            merges.append([best_pair[0], best_pair[1]])

            # Заново разбираем только слова, содержащие победившую пару.
            for word_id in list(pair_words.get(best_pair, ())):
                symbols = splits[word_id]
                frequency = frequencies[word_id]

                for pair in zip(symbols, symbols[1:]):
                    previous = pair_counts[pair]
                    updated = previous - frequency
                    pair_counts[pair] = updated
                    pair_words[pair].discard(word_id)
                    if updated != previous:
                        buckets.get(previous, set()).discard(pair)
                        if updated > 0:
                            buckets.setdefault(updated, set()).add(pair)

                merged_symbols = self._merge_pair(symbols, best_pair)
                splits[word_id] = merged_symbols

                for pair in zip(merged_symbols, merged_symbols[1:]):
                    previous = pair_counts.get(pair, 0)
                    updated = previous + frequency
                    pair_counts[pair] = updated
                    pair_words.setdefault(pair, set()).add(word_id)
                    if updated != previous:
                        if previous > 0:
                            buckets.get(previous, set()).discard(pair)
                        buckets.setdefault(updated, set()).add(pair)

        self.merges = merges
        self.merge_ranks = {tuple(pair): rank for rank, pair in enumerate(merges)}
        self._cache = {}
        return self

    def apply(self, word: str) -> list[str]:
        """Токенизирует ``word``, последовательно применяя пару с наименьшим рангом.

        Результаты кэшируются по типу слова, потому что одно и то же слово
        встречается в корпусе много раз.
        """
        cached = self._cache.get(word)
        if cached is not None:
            return list(cached)

        symbols = list(word) + [END_OF_WORD]
        ranks = self.merge_ranks
        while len(symbols) > 1:
            best_rank = None
            best_pair = None
            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                rank = ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            symbols = self._merge_pair(symbols, best_pair)

        result = tuple(symbols)
        self._cache[word] = result
        return list(result)

    def corpus_statistics(
        self,
        word_freqs: Counter,
        example_words: list[str],
    ) -> dict:
        """Применяет токенизатор ко всем типам слов и подводит итоги сжатия.

        ``tokens_bpe`` — суммарное число BPE-токенов в корпусе,
        ``tokens_per_word`` — в среднем на словоупотребление,
        ``compression_vs_chars`` — во сколько раз токенов меньше, чем символов.
        """
        token_vocabulary: set[str] = set()
        total_chars = 0
        total_words = 0
        total_tokens = 0
        examples: dict[str, list[str]] = {}

        for word, frequency in word_freqs.items():
            tokens = self.apply(word)
            token_vocabulary.update(tokens)
            total_chars += len(word) * frequency
            total_words += frequency
            total_tokens += len(tokens) * frequency

        for word in example_words:
            examples[word] = self.apply(word)

        return {
            "num_merges": len(self.merges),
            "vocab_size": len(token_vocabulary),
            "merges": self.merges,
            "stats": {
                "chars": total_chars,
                "words": total_words,
                "tokens_bpe": total_tokens,
                "tokens_per_word": total_tokens / total_words,
                "compression_vs_chars": total_chars / total_tokens,
                "examples": examples,
            },
        }


# ---------------------------------------------------------------------------
# Сборка артефактов
# ---------------------------------------------------------------------------


def build_bigram_artifact(
    words: list[str],
    train_words: list[str],
    val_words: list[str],
    train_counts: Counter,
) -> dict:
    """Обучает модель, оценивает её и собирает содержимое ``bigram.json``."""
    # Модель, по которой считаются все числа, сглажена при alpha=0.01.
    model = BigramModel(alpha=LIKELIHOOD_ALPHA).fit(train_words)

    perplexities = {
        f"{alpha:g}": perplexity(model, val_words, alpha)
        for alpha in PERPLEXITY_ALPHAS
    }

    sentence = val_words[:12]
    shuffled = list(sentence)
    shuffler = random.Random(SHUFFLE_SEED)
    for _ in range(100):
        shuffler.shuffle(shuffled)
        if shuffled != sentence:
            break

    likelihood = {
        "sentence": sentence,
        "shuffled": shuffled,
        "alpha": LIKELIHOOD_ALPHA,
        "logprob_sentence": model.logprob(sentence, LIKELIHOOD_ALPHA),
        "logprob_shuffled": model.logprob(shuffled, LIKELIHOOD_ALPHA),
        "unseen_pairs_in_sentence": count_unseen_pairs(model, sentence),
    }

    generator = random.Random(GENERATION_SEED)
    generation = []
    for temperature in GENERATION_TEMPERATURES:
        for start in GENERATION_STARTS:
            sample = model.generate(start, GENERATION_LENGTH, temperature, generator)
            generation.append(
                {
                    "temperature": temperature,
                    "start": start,
                    "text": sample,
                }
            )

    return {
        "corpus": CORPUS_URL,
        "words": len(words),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "vocab_size": model.vocab_size,
        "top20": most_common_pairs(train_counts),
        "perplexity": perplexities,
        "likelihood": likelihood,
        "generation": generation,
    }


def save_json(path: Path, payload: dict) -> None:
    """Записывает ``payload`` в UTF-8 JSON в стабильном удобочитаемом виде."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[артефакты] записан {path}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает флаги командной строки, задающие длину прогона и размер BPE."""
    parser = argparse.ArgumentParser(
        description="Биграммная языковая модель и BPE-токенизатор для «Войны и мира»."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="быстрый прогон: 200 слияний BPE и более высокий порог частоты",
    )
    parser.add_argument(
        "--merges",
        type=int,
        default=None,
        help="переопределить число слияний BPE (например, 300 или 5000)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=None,
        help="переопределить минимальную частоту слов при обучении BPE",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Выполняет весь конвейер и записывает оба JSON-артефакта."""
    started = time.perf_counter()
    args = parse_args(argv)

    num_merges = args.merges
    if num_merges is None:
        num_merges = FAST_MERGES if args.fast else FULL_MERGES
    min_frequency = args.min_frequency
    if min_frequency is None:
        min_frequency = FAST_MIN_FREQUENCY if args.fast else FULL_MIN_FREQUENCY

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    download_corpus(CORPUS_URL, CORPUS_PATH)
    words = load_words(CORPUS_PATH)
    if not words:
        print("[корпус] слова не найдены, прерываю", file=sys.stderr)
        return 1
    print(f"[корпус] слов: {len(words)}, типов: {len(set(words))}")

    train_words, val_words = split_train_val(words, VAL_FRACTION)
    train_counts = Counter(train_words)
    full_counts = Counter(words)
    print(
        f"[корпус] обучение={len(train_words)} проверка={len(val_words)} "
        f"({len(val_words) / len(words):.2%}) словарь(обучение)={len(train_counts)}"
    )

    bigram_payload = build_bigram_artifact(words, train_words, val_words, train_counts)
    save_json(ARTIFACTS_DIR / "bigram.json", bigram_payload)

    tokenizer = BPETokenizer(min_frequency=min_frequency)
    tokenizer.fit(full_counts, num_merges)
    print(f"[BPE] выучено слияний: {len(tokenizer.merges)}")

    example_words = [
        "наполеон",
        "сказала",
        "неудовольствие",
        "князь",
        "андрей",
        "противный",
    ]
    bpe_payload = tokenizer.corpus_statistics(full_counts, example_words)
    save_json(ARTIFACTS_DIR / "bpe.json", bpe_payload)

    elapsed = time.perf_counter() - started
    print(f"[готово] выполнено за {elapsed:.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

