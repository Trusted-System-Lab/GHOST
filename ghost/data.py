"""Dataset loading utilities for the classification transformation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from datasets import load_dataset


@dataclass(frozen=True)
class TextRecord:
    text: str
    label: int | str | None = None


def _dataset_columns(name: str) -> tuple[str, str, str | None]:
    aliases = {
        "tweet": "tweeter",
        "tweets": "tweeter",
        "tweet_sentiment": "tweeter",
        "tweet_sentiment_extraction": "tweeter",
    }
    name = aliases.get(name, name)
    if name in {"cola", "sst2"}:
        return "glue", "sentence", name
    if name == "rotten_tomatoes":
        return "rotten_tomatoes", "text", None
    if name == "tweeter":
        return "SetFit/tweet_sentiment_extraction", "text", None
    if name == "yahoo":
        return "yahoo_answers_topics", "question_title", None
    raise ValueError(
        f"Unsupported dataset {name!r}. Supported values: cola, sst2, "
        "rotten_tomatoes, tweeter, yahoo."
    )


def _balanced_indices(labels: Iterable[int | str], total: int, seed: int) -> list[int]:
    labels = list(labels)
    label_to_indices: dict[int | str, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    class_labels = sorted(label_to_indices, key=lambda value: str(value))
    base = total // len(class_labels)
    remainder = total % len(class_labels)

    selected: list[int] = []
    for class_offset, label in enumerate(class_labels):
        target = base + (1 if class_offset < remainder else 0)
        candidates = label_to_indices[label]
        if target <= len(candidates):
            selected.extend(rng.sample(candidates, target))
        else:
            # Small debug subsets can request more examples than a class has.
            # Sampling with replacement keeps the class balance explicit.
            selected.extend(rng.choices(candidates, k=target))

    rng.shuffle(selected)
    return selected


def load_text_classification_data(
    dataset_name: str,
    *,
    num_samples: int,
    seed: int = 42,
    split: str = "train",
) -> list[TextRecord]:
    """Load and class-balance one of the datasets used by the paper.

    The old scripts sampled with replacement even when enough examples were
    available. Here we sample without replacement first, then fall back to
    replacement only when the requested class quota exceeds the available data.
    """

    hf_name, text_column, glue_config = _dataset_columns(dataset_name)
    dataset = load_dataset(hf_name, glue_config, split=split) if glue_config else load_dataset(hf_name, split=split)

    label_column = "label" if "label" in dataset.column_names else "topic"
    labels = list(dataset[label_column])
    texts = list(dataset[text_column])

    if num_samples <= 0:
        selected = list(range(len(texts)))
    else:
        selected = _balanced_indices(labels, num_samples, seed)

    return [TextRecord(text=str(texts[idx]), label=labels[idx]) for idx in selected]


GENERATION_DATA_FILES = {
    "enron": "enron.json",
    "medical": "medical_train_data.json",
    "legal": "legal_train_data.json",
    "news": "news_train_data.json",
    "fine_persona": "fine_persona_train_data.json",
    "medical_chatbot": "medical_chatbot_train_data.json",
    "medical_qna": "medical_qna_train_data.json",
    "legal_task": "legal_task_train_data.json",
    "news_dataset": "news_dataset_train_data.json",
}


def is_classification_dataset(dataset_name: str) -> bool:
    try:
        _dataset_columns(dataset_name)
    except ValueError:
        return False
    return True


def is_generation_dataset(dataset_name: str) -> bool:
    return dataset_name in GENERATION_DATA_FILES


def load_text_generation_data(
    dataset_name: str,
    *,
    num_samples: int,
    seed: int = 42,
    data_dir: str | Path = "../data",
    max_words: int | None = None,
) -> list[TextRecord]:
    """Load local generation datasets used by the original scripts.

    These datasets were already preprocessed into JSON sentence lists in the old
    project. We keep that contract, but make the data directory explicit so the
    reconstructed repo can live separately from the original experiment outputs.
    """

    if dataset_name not in GENERATION_DATA_FILES:
        raise ValueError(
            f"Unsupported generation dataset {dataset_name!r}. Supported values: "
            f"{', '.join(sorted(GENERATION_DATA_FILES))}."
        )

    path = Path(data_dir) / GENERATION_DATA_FILES[dataset_name]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    sentences = [str(item) for item in payload]
    if max_words is not None:
        sentences = [sentence for sentence in sentences if len(sentence.split()) <= max_words]

    rng = random.Random(seed)
    rng.shuffle(sentences)
    if num_samples > 0:
        sentences = sentences[:num_samples]

    return [TextRecord(text=sentence) for sentence in sentences]


def model_output_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]
