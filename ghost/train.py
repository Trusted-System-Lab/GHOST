"""Training and utility evaluation for original and GHOST-obfuscated data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Literal

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .data import is_classification_dataset, is_generation_dataset, model_output_name
from .pipeline import resolve_device


TrainSource = Literal["original", "transformed"]
TaskName = Literal["auto", "classification", "generation"]
LoraMode = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    model_name: str
    task: TaskName = "auto"
    train_source: TrainSource = "transformed"
    transformed_data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "models"
    top_k: int = 70
    beam_width: int = 1
    overlap_threshold: float = 0.1
    num_samples: int = 1000
    train_ratio: float = 0.8
    batch_size: int = 32
    learning_rate: float = 1e-5
    patience: int = 5
    max_epochs: int = 100
    seed: int = 42
    device: str = "auto"
    max_length: int | None = None
    hf_token: str | None = None
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    lora: LoraMode = "auto"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    gradient_noise: float | None = None
    gradient_prune: float | None = None
    save_model: bool = True
    overwrite_results: bool = True


class DictTensorDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor]):
        self.encodings = encodings

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.encodings.items()}

    def __len__(self) -> int:
        return int(next(iter(self.encodings.values())).shape[0])


def transformed_data_path(config: TrainConfig) -> Path:
    model_dir = model_output_name(config.model_name)
    filename = (
        f"{config.dataset}_top_{config.top_k}_beam_{config.beam_width}"
        f"_overlap_{config.overlap_threshold}_discrete_transformed_data.json"
    )
    return Path(config.transformed_data_dir) / model_dir / filename


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _task_from_config(config: TrainConfig) -> Literal["classification", "generation"]:
    if config.task == "classification":
        return "classification"
    if config.task == "generation":
        return "generation"
    if is_classification_dataset(config.dataset):
        return "classification"
    if is_generation_dataset(config.dataset):
        return "generation"
    raise ValueError(f"Could not infer task for dataset {config.dataset!r}; pass --task explicitly.")


def _hf_token(config: TrainConfig) -> str | None:
    return config.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _torch_dtype(dtype_name: str) -> torch.dtype | str:
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


def _is_lora_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "llama" in lowered or "gemma" in lowered


def _should_use_lora(config: TrainConfig, task: str) -> bool:
    if task != "generation":
        return False
    if config.lora == "on":
        return True
    if config.lora == "off":
        return False
    return _is_lora_model(config.model_name)


def _load_transformation_payload(config: TrainConfig) -> dict[str, Any]:
    path = transformed_data_path(config)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def _load_texts_and_labels(config: TrainConfig, task: str) -> tuple[list[str], list[str], list[int] | None]:
    payload = _load_transformation_payload(config)
    original = [str(item) for item in payload["original_sentences"]]
    transformed = [str(item) for item in payload.get("transformed_sentences", [])]
    labels = payload.get("labels")

    if config.train_source == "transformed":
        if len(transformed) < len(original):
            raise ValueError(
                f"Transformed data is incomplete: {len(transformed)} transformed samples "
                f"for {len(original)} original samples."
            )
        train_texts = transformed
    else:
        train_texts = original

    if config.num_samples > 0:
        train_texts = train_texts[: config.num_samples]
        original = original[: config.num_samples]
        labels = labels[: config.num_samples] if labels is not None else None

    if task == "classification" and labels is None:
        raise ValueError("Classification training requires labels in the transformed data JSON.")

    return train_texts, original, labels


def _shuffle_split(
    train_texts: list[str],
    eval_texts: list[str],
    labels: list[int] | None,
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[int] | None, list[int] | None]:
    if len(train_texts) != len(eval_texts):
        raise ValueError("Training and evaluation text lists must have the same length.")

    indices = list(range(len(train_texts)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(len(indices) * train_ratio)

    train_indices = indices[:split]
    eval_indices = indices[split:]

    train_split = [train_texts[idx] for idx in train_indices]
    eval_split = [eval_texts[idx] for idx in eval_indices]

    if labels is None:
        return train_split, eval_split, None, None
    return train_split, eval_split, [int(labels[idx]) for idx in train_indices], [int(labels[idx]) for idx in eval_indices]


def _classification_datasets(
    tokenizer: Any,
    train_texts: list[str],
    eval_texts: list[str],
    train_labels: list[int],
    eval_labels: list[int],
    max_length: int | None,
) -> tuple[DictTensorDataset, DictTensorDataset]:
    train = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    train["labels"] = torch.tensor(train_labels, dtype=torch.long)
    eval_set = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    eval_set["labels"] = torch.tensor(eval_labels, dtype=torch.long)
    return DictTensorDataset(train), DictTensorDataset(eval_set)


def _causal_lm_datasets(
    tokenizer: Any,
    train_texts: list[str],
    eval_texts: list[str],
    max_length: int | None,
) -> tuple[DictTensorDataset, DictTensorDataset]:
    train = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    eval_set = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Pad tokens are not part of the language-model target. The old scripts
    # cloned input_ids directly, which accidentally trained on padding.
    train_labels = train["input_ids"].clone()
    train_labels[train["attention_mask"] == 0] = -100
    eval_labels = eval_set["input_ids"].clone()
    eval_labels[eval_set["attention_mask"] == 0] = -100
    train["labels"] = train_labels
    eval_set["labels"] = eval_labels
    return DictTensorDataset(train), DictTensorDataset(eval_set)


def _model_kwargs(config: TrainConfig, device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    token = _hf_token(config)
    if token:
        kwargs["token"] = token
    dtype = _torch_dtype(config.torch_dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype

    if config.load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit bitsandbytes loading requires a CUDA device.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    return kwargs


def _load_tokenizer(config: TrainConfig, task: str) -> Any:
    kwargs: dict[str, Any] = {"use_fast": True}
    token = _hf_token(config)
    if token:
        kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, **kwargs)
    if task == "generation":
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_classification_model(config: TrainConfig, num_labels: int, device: torch.device) -> Any:
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        **_model_kwargs(config, device),
    )
    if not config.load_in_4bit:
        model.to(device)
    return model


def _load_generation_model(config: TrainConfig, tokenizer: Any, device: torch.device, use_lora: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **_model_kwargs(config, device))
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if not config.load_in_4bit:
        model.to(device)

    if use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    return model


def _apply_gradient_defense(model: torch.nn.Module, config: TrainConfig, device: torch.device) -> None:
    if config.gradient_noise is None and config.gradient_prune is None:
        return
    if config.gradient_noise is not None and config.gradient_prune is not None:
        raise ValueError("Use either gradient noise or gradient pruning, not both.")

    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if config.gradient_noise is not None:
            grad_norm = param.grad.norm()
            if grad_norm > 0:
                param.grad = (param.grad / grad_norm) + torch.randn_like(param.grad, device=device) * config.gradient_noise
        elif config.gradient_prune is not None:
            keep_mask = (torch.rand_like(param.grad, device=device) > config.gradient_prune).to(param.grad.dtype)
            param.grad = param.grad * keep_mask


def _classification_metrics(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    preds: list[int] = []
    gold: list[int] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        losses.append(float(output.loss.item()))
        predictions = torch.argmax(output.logits, dim=-1)
        preds.extend(predictions.detach().cpu().tolist())
        gold.extend(batch["labels"].detach().cpu().tolist())

    accuracy = sum(int(pred == label) for pred, label in zip(preds, gold, strict=True)) / max(1, len(gold))
    average = "binary" if len(set(gold)) == 2 else "macro"
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": accuracy,
        "f1": float(f1_score(gold, preds, average=average)),
    }


def _generation_metrics(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        losses.append(float(output.loss.item()))
    loss = sum(losses) / max(1, len(losses))
    perplexity = math.exp(loss) if loss < 709 else math.inf
    return {"loss": loss, "perplexity": float(perplexity)}


def _train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        _apply_gradient_defense(model, config, device)
        optimizer.step()
        losses.append(float(output.loss.item()))
    return sum(losses) / max(1, len(losses))


def _result_file(config: TrainConfig, task: str) -> Path:
    model_dir = model_output_name(config.model_name)
    suffix = config.train_source
    if config.gradient_noise is not None:
        suffix += f"_noise_{config.gradient_noise}"
    if config.gradient_prune is not None:
        suffix += f"_prune_{config.gradient_prune}"
    return Path(config.results_dir) / model_dir / f"{config.dataset}_{suffix}_{task}_metrics.json"


def _model_output_dir(config: TrainConfig) -> Path:
    model_dir = model_output_name(config.model_name)
    suffix = config.train_source
    if config.gradient_noise is not None:
        suffix += f"_noise_{config.gradient_noise}"
    if config.gradient_prune is not None:
        suffix += f"_prune_{config.gradient_prune}"
    return Path(config.models_dir) / model_dir / f"{config.dataset}_{suffix}"


def run_training(config: TrainConfig) -> Path:
    """Train a model and write per-epoch utility metrics to JSON."""

    _set_seed(config.seed)
    task = _task_from_config(config)
    device = resolve_device(config.device)
    train_texts, eval_texts, labels = _load_texts_and_labels(config, task)
    train_texts, eval_texts, train_labels, eval_labels = _shuffle_split(
        train_texts,
        eval_texts,
        labels,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )

    tokenizer = _load_tokenizer(config, task)
    use_lora = _should_use_lora(config, task)
    if task == "classification":
        assert train_labels is not None and eval_labels is not None
        model = _load_classification_model(config, num_labels=len(set(labels or [])), device=device)
        train_dataset, eval_dataset = _classification_datasets(
            tokenizer,
            train_texts,
            eval_texts,
            train_labels,
            eval_labels,
            config.max_length,
        )
        metric_fn = _classification_metrics
        monitor_key = "f1"
        larger_is_better = True
    else:
        model = _load_generation_model(config, tokenizer, device=device, use_lora=use_lora)
        train_dataset, eval_dataset = _causal_lm_datasets(tokenizer, train_texts, eval_texts, config.max_length)
        metric_fn = _generation_metrics
        monitor_key = "perplexity"
        larger_is_better = False

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=config.learning_rate)

    result_path = _result_file(config, task)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and config.overwrite_results:
        result_path.unlink()

    model_dir = _model_output_dir(config)
    best_metric = -math.inf if larger_is_better else math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    initial_eval = metric_fn(model, eval_loader, device)
    history.append({"epoch": 0, "phase": "initial_eval", "eval": initial_eval})
    _write_metrics(result_path, config, task, use_lora, best_epoch, history)

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, config, device)
        eval_metrics = metric_fn(model, eval_loader, device)
        current = eval_metrics[monitor_key]
        improved = current > best_metric if larger_is_better else current < best_metric

        if improved:
            best_metric = current
            best_epoch = epoch
            stale_epochs = 0
            if config.save_model:
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "phase": "train_eval",
                "train": {"loss": train_loss},
                "eval": eval_metrics,
                "best_epoch": best_epoch,
            }
        )
        _write_metrics(result_path, config, task, use_lora, best_epoch, history)

        if stale_epochs >= config.patience:
            break

    return result_path


def _write_metrics(
    path: Path,
    config: TrainConfig,
    task: str,
    use_lora: bool,
    best_epoch: int,
    history: list[dict[str, Any]],
) -> None:
    payload = {
        "config": asdict(config),
        "task": task,
        "uses_lora": use_lora,
        "best_epoch": best_epoch,
        "history": history,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
