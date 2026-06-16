import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NeuronResult:
    layer_index: int
    neuron_index: int
    max_activation: float
    mean_activation: float
    std_activation: float
    fraction_active: float
    activating_token: str
    activating_token_position: int
    context_window: list[str]
    context_window_positions: list[int]
    activation_values_per_token: list[float]
    is_dead: bool
    z_score: float = 0.0
    rank: int = 0


@dataclass
class LayerSummary:
    layer_index: int
    total_neurons: int
    dead_neurons: int
    max_activation: float
    mean_activation: float
    fraction_dead: float


class NeuronAnalyser:
    """Analyses MLP neuron activations across all layers.

    Takes captured MLP activation tensors and token strings, then computes
    per-neuron statistics, ranks neurons by activation strength, and extracts
    context windows for explanation generation.
    """

    def __init__(
        self,
        mlp_activations: list,
        tokens: list[str],
        context_window_size: int = 5,
    ):
        self.mlp_activations = mlp_activations
        self.tokens = tokens
        self.context_window_size = context_window_size
        self.seq_len = len(tokens)
        self._neuron_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_neuron_activation_vector(self, layer_index: int, neuron_index: int) -> torch.Tensor:
        """Get the activation vector across all token positions for a single neuron.

        Returns a 1D tensor of shape [seq_len] with the activation value of
        this neuron at each token position.
        """
        cache_key = (layer_index, neuron_index)
        if cache_key in self._neuron_cache:
            return self._neuron_cache[cache_key]

        capture = self.mlp_activations[layer_index]
        output = capture.output_tensor  # [batch, seq_len, d_mlp] or [seq_len, d_mlp]

        if output.dim() == 3:
            output = output[0]
        neuron_acts = output[:, neuron_index].float()

        self._neuron_cache[cache_key] = neuron_acts
        return neuron_acts

    def compute_neuron_stats(self, layer_index: int, neuron_index: int) -> dict:
        """Compute activation statistics for a single neuron on this prompt."""
        acts = self._get_neuron_activation_vector(layer_index, neuron_index)
        acts_np = acts.detach().numpy()

        max_act = float(acts_np.max())
        mean_act = float(acts_np.mean())
        std_act = float(acts_np.std()) if acts_np.std() > 0 else 0.0
        frac_active = float((acts_np > 0).mean())
        is_dead = max_act == 0.0

        return {
            "max_activation": max_act,
            "mean_activation": mean_act,
            "std_activation": std_act,
            "fraction_active": frac_active,
            "is_dead": is_dead,
        }

    def get_top_activating_position(self, layer_index: int, neuron_index: int) -> tuple[int, float]:
        """Get the token position and value of the maximum activation."""
        acts = self._get_neuron_activation_vector(layer_index, neuron_index)
        pos = int(acts.argmax().item())
        val = float(acts[pos].item())
        return pos, val

    def get_context_window(self, center_position: int) -> tuple[list[str], list[int]]:
        """Extract a window of tokens around a center position."""
        half = self.context_window_size
        start = max(0, center_position - half)
        end = min(self.seq_len, center_position + half + 1)
        window_tokens = self.tokens[start:end]
        window_positions = list(range(start, end))
        return window_tokens, window_positions

    def compute_all_neurons_for_layer(self, layer_index: int) -> list[NeuronResult]:
        """Compute NeuronResult for every neuron in a single layer.

        This is the inner loop of the analysis — it processes all neurons
        across the full MLP hidden dimension for one layer.
        """
        capture = self.mlp_activations[layer_index]
        output = capture.output_tensor
        if output.dim() == 3:
            output = output[0]
        d_mlp = output.shape[-1]

        results = []
        for neuron_idx in range(d_mlp):
            stats = self.compute_neuron_stats(layer_index, neuron_idx)
            pos, max_val = self.get_top_activating_position(layer_index, neuron_idx)
            token = self.tokens[pos] if pos < len(self.tokens) else "[UNK]"
            context_tokens, context_pos = self.get_context_window(pos)

            acts = self._get_neuron_activation_vector(layer_index, neuron_idx)
            activation_values = acts.tolist()

            result = NeuronResult(
                layer_index=layer_index,
                neuron_index=neuron_idx,
                max_activation=stats["max_activation"],
                mean_activation=stats["mean_activation"],
                std_activation=stats["std_activation"],
                fraction_active=stats["fraction_active"],
                activating_token=token,
                activating_token_position=pos,
                context_window=context_tokens,
                context_window_positions=context_pos,
                activation_values_per_token=activation_values,
                is_dead=stats["is_dead"],
                z_score=0.0,
                rank=0,
            )
            results.append(result)

        return results

    def rank_top_k(self, k: int = 20) -> list[NeuronResult]:
        """Rank all neurons across all layers by max activation value.

        Returns the top-K neurons as a list of NeuronResult dataclasses,
        sorted descending by max_activation.
        """
        all_neurons = []
        for layer_idx in range(len(self.mlp_activations)):
            layer_results = self.compute_all_neurons_for_layer(layer_idx)
            all_neurons.extend(layer_results)

        all_neurons.sort(key=lambda r: r.max_activation, reverse=True)

        for i, neuron in enumerate(all_neurons[:k]):
            neuron.rank = i + 1

        return all_neurons[:k]

    def compute_layer_summaries(self) -> list[LayerSummary]:
        """Compute per-layer aggregate statistics."""
        summaries = []
        for layer_idx in range(len(self.mlp_activations)):
            capture = self.mlp_activations[layer_idx]
            output = capture.output_tensor
            if output.dim() == 3:
                output = output[0]

            acts = output.float()

            max_act = float(acts.max().item())
            mean_act = float(acts.mean().item())

            total_neurons = acts.shape[-1]
            dead_mask = (acts == 0).all(dim=0)
            dead_count = int(dead_mask.sum().item())

            summaries.append(LayerSummary(
                layer_index=layer_idx,
                total_neurons=total_neurons,
                dead_neurons=dead_count,
                max_activation=max_act,
                mean_activation=mean_act,
                fraction_dead=dead_count / total_neurons if total_neurons > 0 else 0.0,
            ))

        return summaries

    def compute_neuron_cosine_similarity(
        self, layer_index: int, neuron_index: int, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Find similar neurons in the same layer by cosine similarity of activation vectors."""
        target_acts = self._get_neuron_activation_vector(layer_index, neuron_index)
        target_norm = target_acts / (target_acts.norm() + 1e-8)

        capture = self.mlp_activations[layer_index]
        output = capture.output_tensor
        if output.dim() == 3:
            output = output[0]

        similarities = []
        d_mlp = output.shape[-1]
        for other_idx in range(d_mlp):
            if other_idx == neuron_index:
                continue
            other_acts = output[:, other_idx].float()
            other_norm = other_acts / (other_acts.norm() + 1e-8)
            sim = float(torch.dot(target_norm, other_norm).item())
            similarities.append((other_idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_dead_neuron_count(self, layer_index: int) -> int:
        """Count dead (zero-activation) neurons in a given layer."""
        capture = self.mlp_activations[layer_index]
        output = capture.output_tensor
        if output.dim() == 3:
            output = output[0]
        dead_mask = (output == 0).all(dim=0)
        return int(dead_mask.sum().item())

    def get_activation_heatmap(self, layer_index: int, neuron_index: int) -> list[float]:
        """Get full per-token activation values for a specific neuron.

        Used to drive the per-token activation heatmap in the dashboard.
        """
        acts = self._get_neuron_activation_vector(layer_index, neuron_index)
        return acts.tolist()
