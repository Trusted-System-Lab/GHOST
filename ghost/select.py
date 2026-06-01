"""Hidden-state-preserving shadow-token selection."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SelectionConfig:
    beam_width: int = 1
    early_stop_delta: float = 0.1
    max_iterations: int = 20
    eval_batch_size: int = 64
    include_embedding_layer: bool = True


@dataclass(frozen=True)
class SelectionResult:
    token_ids: list[int]
    loss: float
    iterations: int
    converged: bool


class HiddenStateSelector:
    """Coordinate beam search over shadow-token candidates."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        shadow_map: Sequence[Sequence[int]],
        config: SelectionConfig | None = None,
        rng: random.Random | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.shadow_map = shadow_map
        self.config = config or SelectionConfig()
        self.rng = rng or random.Random()
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []))

    def select(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        original_hidden_states: Sequence[torch.Tensor] | None = None,
    ) -> SelectionResult:
        if input_ids.dim() != 1:
            raise ValueError("HiddenStateSelector expects one tokenized sequence at a time.")

        device = input_ids.device
        attention_mask = attention_mask.to(device)
        if original_hidden_states is None:
            with torch.no_grad():
                original_hidden_states = self.model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    output_hidden_states=True,
                ).hidden_states

        original = self._prepare_original_hidden_states(original_hidden_states)
        mutable_positions = self._mutable_positions(input_ids, attention_mask)
        initial = self._random_initial_solution(input_ids, attention_mask)
        beams = self._rank([initial], original, attention_mask)

        if not mutable_positions:
            return SelectionResult(token_ids=initial, loss=beams[0][1], iterations=0, converged=True)

        converged = False
        iteration = 0
        for iteration in range(1, self.config.max_iterations + 1):
            previous_best = beams[0][1]

            for position in mutable_positions:
                original_token = int(input_ids[position].item())
                candidates = self._candidates_for(original_token)
                if not candidates:
                    continue

                expanded: list[list[int]] = [beam for beam, _ in beams]
                for beam, _ in beams:
                    for candidate in candidates:
                        if candidate == beam[position]:
                            continue
                        proposal = list(beam)
                        proposal[position] = candidate
                        expanded.append(proposal)

                beams = self._rank(expanded, original, attention_mask)

            improvement = previous_best - beams[0][1]
            if improvement <= self.config.early_stop_delta:
                converged = True
                break

        best_tokens, best_loss = beams[0]
        return SelectionResult(
            token_ids=list(best_tokens),
            loss=float(best_loss),
            iterations=iteration,
            converged=converged,
        )

    def _prepare_original_hidden_states(self, hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        layers = list(hidden_states)
        if not self.config.include_embedding_layer and len(layers) > 1:
            layers = layers[1:]
        return [layer.detach() for layer in layers]

    def _mutable_positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[int]:
        positions: list[int] = []
        for idx, (token_id, keep) in enumerate(zip(input_ids.tolist(), attention_mask.tolist(), strict=True)):
            if not keep:
                continue
            if int(token_id) in self.special_ids:
                continue
            positions.append(idx)
        return positions

    def _random_initial_solution(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[int]:
        solution = input_ids.detach().cpu().tolist()
        for position in self._mutable_positions(input_ids, attention_mask):
            token_id = int(input_ids[position].item())
            candidates = self._candidates_for(token_id)
            if candidates:
                solution[position] = self.rng.choice(candidates)
        return solution

    def _candidates_for(self, token_id: int) -> list[int]:
        if token_id < 0 or token_id >= len(self.shadow_map):
            return []
        return [int(candidate) for candidate in self.shadow_map[token_id] if int(candidate) not in self.special_ids]

    def _rank(
        self,
        solutions: Sequence[Sequence[int]],
        original_hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> list[tuple[list[int], float]]:
        unique: dict[tuple[int, ...], list[int]] = {}
        for solution in solutions:
            key = tuple(int(token_id) for token_id in solution)
            unique.setdefault(key, list(key))

        scored: list[tuple[list[int], float]] = []
        values = list(unique.values())
        for start in range(0, len(values), self.config.eval_batch_size):
            chunk = values[start : start + self.config.eval_batch_size]
            losses = self._score(chunk, original_hidden_states, attention_mask)
            scored.extend(zip(chunk, losses, strict=True))

        scored.sort(key=lambda item: item[1])
        return scored[: self.config.beam_width]

    def _score(
        self,
        solutions: Sequence[Sequence[int]],
        original_hidden_states: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> list[float]:
        device = attention_mask.device
        ids = torch.tensor(solutions, dtype=torch.long, device=device)
        repeated_mask = attention_mask.unsqueeze(0).expand(ids.shape[0], -1)

        with torch.no_grad():
            outputs = self.model(input_ids=ids, attention_mask=repeated_mask, output_hidden_states=True)

        candidate_layers = list(outputs.hidden_states)
        if not self.config.include_embedding_layer and len(candidate_layers) > 1:
            candidate_layers = candidate_layers[1:]

        losses = torch.zeros(ids.shape[0], dtype=torch.float32, device=device)
        for original_layer, candidate_layer in zip(original_hidden_states, candidate_layers, strict=True):
            reference = original_layer.to(device)
            if reference.shape[0] == 1:
                reference = reference.expand(candidate_layer.shape[0], -1, -1)
            losses += F.mse_loss(candidate_layer, reference, reduction="none").mean(dim=(1, 2))

        return losses.detach().cpu().tolist()
