"""Shadow-token search for GHOST."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import spacy
import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass(frozen=True)
class ShadowSearchConfig:
    top_k: int = 70
    overlap_threshold: float = 0.1
    expand_step: int = 10
    max_search_k: int | None = None
    similarity_batch_size: int = 512
    exclude_special_candidates: bool = True


def _token_text(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if isinstance(token, list):
        token = token[0]
    return str(token)


def _looks_like_subword(token: str, tokenizer_name: str) -> bool:
    if token.startswith("##"):
        return True
    # RoBERTa/GPT-2 byte-level tokens mark word starts with \u0120.
    if "roberta" in tokenizer_name or "gpt2" in tokenizer_name:
        return not token.startswith("\u0120")
    # SentencePiece models mark word starts with \u2581.
    if token.startswith("\u2581"):
        return False
    return False


def _normalised_piece(token: str) -> str:
    return token.replace("##", "").replace("\u0120", "").replace("\u2581", "").strip().lower()


def _lemma_for_token(tokenizer: Any, nlp: Any, token_id: int) -> str:
    raw_token = _token_text(tokenizer, token_id)
    if int(token_id) in set(getattr(tokenizer, "all_special_ids", [])):
        return f"special:{raw_token.lower()}"

    tokenizer_name = tokenizer.__class__.__name__.lower()
    if _looks_like_subword(raw_token, tokenizer_name):
        return f"piece:{_normalised_piece(raw_token)}"

    surface = tokenizer.convert_tokens_to_string([raw_token]).strip()
    surface = surface or _normalised_piece(raw_token)
    if not any(ch.isalpha() for ch in surface):
        return surface.lower()

    doc = nlp(surface)
    if len(doc) == 1 and doc[0].lemma_:
        return doc[0].lemma_.lower()
    return surface.lower()


def _load_spacy_model() -> Any:
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model en_core_web_sm is required for lemma filtering. "
            "Install it with `uv add <en_core_web_sm wheel URL>` or "
            "`python -m spacy download en_core_web_sm`."
        ) from exc


class ShadowTokenSearcher:
    """Build semantically filtered embedding-neighbor sets.

    The search follows the paper's three filters: indirect similarity via
    neighbor overlap, direct mutual-neighbor similarity, and common lemma.
    """

    def __init__(self, model: Any, tokenizer: Any, config: ShadowSearchConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ShadowSearchConfig()

    def build(self, *, device: torch.device | str = "cpu") -> list[list[int]]:
        embeddings = self.model.get_input_embeddings().weight.detach().to(device=device, dtype=torch.float32)
        vocab_size = embeddings.shape[0]
        initial_k = min(self.config.top_k, vocab_size - 1)
        max_search_k = self.config.max_search_k
        if max_search_k is None:
            max_search_k = self.config.top_k + 10 * self.config.expand_step
        max_search_k = max(initial_k, min(max_search_k, vocab_size - 1))

        top_neighbors = self._top_embedding_neighbors(embeddings, max_search_k)
        lemmas = self._build_lemma_cache(vocab_size)

        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        shadow_map: list[list[int]] = []
        for token_id in tqdm(range(vocab_size), desc="Filtering shadow tokens"):
            if token_id in special_ids:
                shadow_map.append([token_id])
                continue

            candidates = self._filtered_candidates(token_id, top_neighbors, lemmas, special_ids, initial_k, max_search_k)
            if not candidates:
                # Keep the pipeline total: if every semantic filter removes all
                # candidates, fall back to nearest non-special neighbors.
                candidates = [
                    candidate
                    for candidate in top_neighbors[token_id][: self.config.top_k]
                    if candidate not in special_ids
                ]
            shadow_map.append(candidates or [token_id])

        return shadow_map

    def _top_embedding_neighbors(self, embeddings: torch.Tensor, max_search_k: int) -> list[list[int]]:
        normalized = F.normalize(embeddings, p=2, dim=1)
        vocab_size = normalized.shape[0]
        all_indices: list[list[int]] = []

        for start in tqdm(range(0, vocab_size, self.config.similarity_batch_size), desc="Searching embeddings"):
            end = min(start + self.config.similarity_batch_size, vocab_size)
            sims = normalized[start:end] @ normalized.T
            row_ids = torch.arange(end - start, device=sims.device)
            col_ids = torch.arange(start, end, device=sims.device)
            sims[row_ids, col_ids] = -torch.inf
            _, indices = torch.topk(sims, k=max_search_k, dim=1)
            all_indices.extend(indices.cpu().tolist())

        return all_indices

    def _build_lemma_cache(self, vocab_size: int) -> list[str]:
        nlp = _load_spacy_model()
        return [
            _lemma_for_token(self.tokenizer, nlp, token_id)
            for token_id in tqdm(range(vocab_size), desc="Lemmatizing vocab")
        ]

    def _filtered_candidates(
        self,
        token_id: int,
        top_neighbors: Sequence[Sequence[int]],
        lemmas: Sequence[str],
        special_ids: set[int],
        initial_k: int,
        max_search_k: int,
    ) -> list[int]:
        for current_k in range(initial_k, max_search_k + 1, self.config.expand_step):
            base = list(top_neighbors[token_id][:current_k])
            base_set = set(base)
            accepted: list[int] = []

            for candidate in base:
                if self.config.exclude_special_candidates and candidate in special_ids:
                    continue

                candidate_neighbors = set(top_neighbors[candidate][:current_k])
                overlap_ratio = len(base_set.intersection(candidate_neighbors)) / float(current_k)
                indirect_similarity = overlap_ratio > self.config.overlap_threshold
                direct_similarity = token_id in candidate_neighbors
                common_lemma = lemmas[token_id] == lemmas[candidate]

                if not (indirect_similarity or direct_similarity or common_lemma):
                    accepted.append(int(candidate))

            if accepted:
                return accepted
        return []


def save_shadow_map(path: str | Path, shadow_map: Sequence[Sequence[int]], metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "shadow_map": shadow_map}, handle)


def load_shadow_map(path: str | Path) -> tuple[list[list[int]], dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["shadow_map"], payload.get("metadata", {})


def search_metadata(model_name: str, config: ShadowSearchConfig) -> dict[str, Any]:
    return {"model_name": model_name, "search": asdict(config)}
