"""End-to-end data transformation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .data import (
    TextRecord,
    is_classification_dataset,
    is_generation_dataset,
    load_text_classification_data,
    load_text_generation_data,
    model_output_name,
)
from .search import ShadowSearchConfig, ShadowTokenSearcher, load_shadow_map, save_shadow_map, search_metadata
from .select import HiddenStateSelector, SelectionConfig


@dataclass(frozen=True)
class TransformConfig:
    dataset: str = "sst2"
    model_name: str = "bert-base-uncased"
    task: str = "auto"
    num_samples: int = 1000
    output_dir: str = "data"
    source_data_dir: str = "../data"
    device: str = "auto"
    seed: int = 42
    max_length: int | None = None
    max_words: int | None = None
    recover_batch: int = 0
    hf_token: str | None = None
    add_eos_token: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "auto"
    overwrite_shadow_cache: bool = False
    search: ShadowSearchConfig = ShadowSearchConfig()
    selection: SelectionConfig = SelectionConfig()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def default_output_path(config: TransformConfig) -> Path:
    model_dir = model_output_name(config.model_name)
    filename = (
        f"{config.dataset}_top_{config.search.top_k}_beam_{config.selection.beam_width}"
        f"_overlap_{config.search.overlap_threshold}_discrete_transformed_data.json"
    )
    return Path(config.output_dir) / model_dir / filename


def default_shadow_cache_path(config: TransformConfig) -> Path:
    model_dir = model_output_name(config.model_name)
    filename = f"shadow_top_{config.search.top_k}_overlap_{config.search.overlap_threshold}.json"
    return Path(config.output_dir) / model_dir / filename


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype {dtype_name!r}.")
    return mapping[dtype_name]


def _token_from_config(config: TransformConfig) -> str | None:
    return config.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _is_generative_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(marker in lowered for marker in ("gpt", "llama", "gemma", "mistral", "qwen"))


def _load_model_and_tokenizer(config: TransformConfig, device: torch.device) -> tuple[Any, Any]:
    token = _token_from_config(config)
    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    if token:
        tokenizer_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, **tokenizer_kwargs)

    if config.add_eos_token and hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = True

    if _is_generative_model_name(config.model_name):
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    dtype = _resolve_torch_dtype(config.torch_dtype)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if token:
        model_kwargs["token"] = token

    if config.load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit bitsandbytes loading requires a CUDA device.")
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    model = AutoModel.from_pretrained(config.model_name, **model_kwargs)
    if not config.load_in_4bit:
        model.to(device)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer


def _load_records(config: TransformConfig) -> tuple[list[TextRecord], str]:
    if config.task not in {"auto", "classification", "generation"}:
        raise ValueError("--task must be one of: auto, classification, generation.")

    if config.task == "classification" or (
        config.task == "auto" and is_classification_dataset(config.dataset)
    ):
        return (
            load_text_classification_data(config.dataset, num_samples=config.num_samples, seed=config.seed),
            "classification",
        )

    if config.task == "generation" or (
        config.task == "auto" and is_generation_dataset(config.dataset)
    ):
        return (
            load_text_generation_data(
                config.dataset,
                num_samples=config.num_samples,
                seed=config.seed,
                data_dir=config.source_data_dir,
                max_words=config.max_words,
            ),
            "generation",
        )

    raise ValueError(
        f"Could not infer task for dataset {config.dataset!r}. Pass --task classification "
        "or --task generation explicitly."
    )


def _load_or_build_shadow_map(
    config: TransformConfig,
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> list[list[int]]:
    cache_path = default_shadow_cache_path(config)
    if cache_path.exists() and not config.overwrite_shadow_cache:
        shadow_map, _ = load_shadow_map(cache_path)
        return shadow_map

    searcher = ShadowTokenSearcher(model, tokenizer, config.search)
    # Similarity search benefits from accelerators, but keeping this explicit
    # makes CPU-only reconstruction runs predictable.
    shadow_map = searcher.build(device=device)
    save_shadow_map(cache_path, shadow_map, search_metadata(config.model_name, config.search))
    return shadow_map


def _load_existing_output(path: Path, recover_batch: int) -> list[str]:
    if recover_batch <= 0:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Cannot recover from batch {recover_batch}; output file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    transformed = list(payload.get("transformed_sentences", []))
    if len(transformed) < recover_batch:
        raise ValueError(
            f"Cannot recover from batch {recover_batch}; {path} only contains "
            f"{len(transformed)} transformed samples."
        )
    return transformed[:recover_batch]


def _write_output(
    path: Path,
    *,
    records: list[TextRecord],
    transformed_sentences: list[str],
    config: TransformConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": asdict(config),
        "transformed_sentences": transformed_sentences,
        "original_sentences": [record.text for record in records],
    }
    if any(record.label is not None for record in records):
        payload["labels"] = [record.label for record in records]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_transform(config: TransformConfig) -> Path:
    """Run the full GHOST transformation pipeline and return the output path."""

    _set_global_seed(config.seed)
    device = resolve_device(config.device)
    records, task = _load_records(config)
    model, tokenizer = _load_model_and_tokenizer(config, device)
    shadow_map = _load_or_build_shadow_map(config, model=model, tokenizer=tokenizer, device=device)

    output_path = default_output_path(config)
    transformed = _load_existing_output(output_path, config.recover_batch)
    start_index = len(transformed)

    encodings = tokenizer(
        [record.text for record in records],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length,
    )
    if task == "generation" and tokenizer.pad_token_id is None:
        raise RuntimeError("Generation tokenization requires a pad token; set tokenizer.pad_token first.")

    for idx in tqdm(range(start_index, len(records)), desc="Transforming sentences"):
        input_ids = encodings["input_ids"][idx].to(device)
        attention_mask = encodings["attention_mask"][idx].to(device)

        with torch.no_grad():
            original_hidden_states = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                output_hidden_states=True,
            ).hidden_states

        selector = HiddenStateSelector(
            model,
            tokenizer,
            shadow_map,
            config.selection,
            rng=random.Random(config.seed + idx),
        )
        result = selector.select(
            input_ids=input_ids,
            attention_mask=attention_mask,
            original_hidden_states=original_hidden_states,
        )
        transformed_text = tokenizer.decode(
            result.token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        transformed.append(transformed_text)
        _write_output(output_path, records=records, transformed_sentences=transformed, config=config)

    return output_path
