import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttentionHeadResult:
    layer_index: int
    head_index: int
    focus_score: float
    entropy: float
    pattern_type: str
    attention_matrix: list[list[float]]
    top_attended_pairs: list[dict]
    is_induction_head: bool
    max_attention_weight: float
    rank: int = 0


def _compute_row_entropy(row_probs: np.ndarray) -> float:
    """Compute entropy of a single attention probability distribution.

    Low entropy (< ~2) means the head attends to few positions (focused).
    High entropy (> ~3) means the head attends diffusely.
    """
    row_probs = np.clip(row_probs, 1e-10, 1.0)
    return float(-np.sum(row_probs * np.log(row_probs)))


def _classify_pattern(attention_matrix: np.ndarray, seq_len: int) -> str:
    """Classify an attention head's pattern type.

    Classification rules:
    - diagonal: most queries attend to their own position
    - previous_token: most queries attend to token-1
    - first_token: most queries attend to position 0
    - last_token: most queries attend to the most recent token(s)
    - diffuse: high entropy, no clear structure
    - content_based: variable pattern depending on content (most interesting)
    """
    if seq_len <= 2:
        return "content_based"

    diag_sum = 0
    prev_sum = 0
    first_sum = 0
    last_sum = 0

    for i in range(seq_len):
        row = attention_matrix[i]
        diag_sum += row[i] if i < len(row) else 0
        prev_sum += row[i - 1] if i > 0 else 0
        first_sum += row[0]
        last_sum += row[-1]

    total = seq_len
    diag_frac = diag_sum / total
    prev_frac = prev_sum / total
    first_frac = first_sum / total
    last_frac = last_sum / total

    entropies = [_compute_row_entropy(attention_matrix[i]) for i in range(seq_len)]
    mean_entropy = np.mean(entropies)
    max_entropy = np.log(seq_len)

    if mean_entropy > 0.9 * max_entropy:
        return "diffuse"

    if diag_frac > 0.5:
        return "diagonal"
    if prev_frac > 0.5:
        return "previous_token"
    if first_frac > 0.5:
        return "first_token"
    if last_frac > 0.5:
        return "last_token"

    return "content_based"


def _detect_induction_head(
    attention_matrix: np.ndarray,
    tokens: list[str],
    threshold: float = 0.3,
) -> bool:
    """Detect if a head is likely an induction head.

    Induction heads exhibit the [A][B]...[A][B] pattern: they attend strongly
    from a token to the token immediately after the previous occurrence of the
    same token.

    Simplified detection: for each position i (i > 1), check if the head
    strongly attends to position i-1 AND the token at position i-1 also appears
    earlier in the sequence.
    """
    if len(tokens) < 4:
        return False

    seq_len = attention_matrix.shape[0]
    induction_score = 0.0
    count = 0

    for i in range(2, seq_len):
        prev_token = tokens[i - 1]
        prev_positions = [j for j in range(i - 1) if tokens[j] == prev_token]
        if not prev_positions:
            continue

        attn_to_prev = attention_matrix[i, i - 1]
        for prev_pos in prev_positions:
            if attn_to_prev > threshold:
                induction_score += attn_to_prev
                count += 1

    if count == 0:
        return False

    avg_score = induction_score / count
    return bool(avg_score > threshold)


def _get_top_attended_pairs(
    attention_matrix: np.ndarray,
    tokens: list[str],
    top_k: int = 10,
) -> list[dict]:
    """Get the top-K (query, key) token pairs by attention weight."""
    seq_len = len(tokens)
    pairs = []

    for i in range(seq_len):
        for j in range(seq_len):
            weight = float(attention_matrix[i, j])
            if weight > 0.01:
                pairs.append({
                    "query_position": i,
                    "key_position": j,
                    "query_token": tokens[i] if i < len(tokens) else "[UNK]",
                    "key_token": tokens[j] if j < len(tokens) else "[UNK]",
                    "weight": weight,
                })

    pairs.sort(key=lambda p: p["weight"], reverse=True)
    return pairs[:top_k]


def compute_focus_score(attention_matrix: np.ndarray) -> float:
    """Compute focus score: 1 - normalised mean entropy.

    A highly focused head (low entropy) gets a score close to 1.0.
    A diffuse head gets a score close to 0.0.
    """
    seq_len = attention_matrix.shape[0]
    if seq_len <= 1:
        return 0.0

    entropies = [_compute_row_entropy(attention_matrix[i]) for i in range(seq_len)]
    mean_entropy = np.mean(entropies)
    max_entropy = np.log(seq_len)

    if max_entropy == 0:
        return 0.0

    return float(max(0.0, 1.0 - mean_entropy / max_entropy))


class AttentionAnalyser:
    """Analyses attention head patterns across all layers.

    Works primarily with native attention weights (from output_attentions=True),
    with fallback to attention output tensor analysis.
    """

    def __init__(
        self,
        attention_activations: list,
        tokens: list[str],
        native_attentions: Optional[list[torch.Tensor]] = None,
        n_heads: int = 12,
    ):
        self.attention_activations = attention_activations
        self.tokens = tokens
        self.native_attentions = native_attentions
        self.n_heads = n_heads
        self.seq_len = len(tokens)

    def _get_attention_matrix(self, layer_index: int, head_index: int) -> Optional[np.ndarray]:
        """Extract the [seq_len, seq_len] attention matrix for a specific head.

        Priority:
        1. Native attentions from output_attentions=True
        2. Attention output tensor analysis (limited)
        3. Returns None if unavailable
        """
        if self.native_attentions is not None and layer_index < len(self.native_attentions):
            attn_weights = self.native_attentions[layer_index]
            if attn_weights.dim() == 4:
                batch_idx = 0
                return attn_weights[batch_idx, head_index].float().detach().numpy()

        return None

    def analyse_head(self, layer_index: int, head_index: int) -> Optional[AttentionHeadResult]:
        """Analyse a single attention head and return its results."""
        attn_matrix = self._get_attention_matrix(layer_index, head_index)

        if attn_matrix is None:
            return None

        focus_score = compute_focus_score(attn_matrix)
        pattern_type = _classify_pattern(attn_matrix, self.seq_len)
        is_induction = _detect_induction_head(attn_matrix, self.tokens)
        top_pairs = _get_top_attended_pairs(attn_matrix, self.tokens)

        seq_len = attn_matrix.shape[0]
        entropies = [_compute_row_entropy(attn_matrix[i]) for i in range(seq_len)]
        mean_entropy = float(np.mean(entropies))
        max_weight = float(attn_matrix.max())

        return AttentionHeadResult(
            layer_index=layer_index,
            head_index=head_index,
            focus_score=focus_score,
            entropy=mean_entropy,
            pattern_type=pattern_type,
            attention_matrix=attn_matrix.tolist(),
            top_attended_pairs=top_pairs,
            is_induction_head=is_induction,
            max_attention_weight=max_weight,

            rank=0,
        )

    def rank_top_k(self, k: int = 10) -> list[AttentionHeadResult]:
        """Rank all attention heads by focus score and return top-K."""
        all_heads = []

        for layer_idx in range(len(self.attention_activations)):
            for head_idx in range(self.n_heads):
                result = self.analyse_head(layer_idx, head_idx)
                if result is not None:
                    all_heads.append(result)

        all_heads.sort(key=lambda h: h.focus_score, reverse=True)

        for i, head in enumerate(all_heads[:k]):
            head.rank = i + 1

        return all_heads[:k]

    def get_induction_heads(self) -> list[AttentionHeadResult]:
        """Return all heads classified as induction heads."""
        results = []
        for layer_idx in range(len(self.attention_activations)):
            for head_idx in range(self.n_heads):
                result = self.analyse_head(layer_idx, head_idx)
                if result is not None and result.is_induction_head:
                    results.append(result)
        return results

    def compute_layer_avg_pattern(self, layer_index: int) -> Optional[np.ndarray]:
        """Compute the average attention pattern across all heads in a layer."""
        if self.native_attentions is None or layer_index >= len(self.native_attentions):
            return None

        attn_weights = self.native_attentions[layer_index]
        if attn_weights.dim() == 4:
            return attn_weights[0].mean(dim=0).float().detach().numpy()
        return None

    def get_pattern_type_distribution(self) -> dict[str, int]:
        """Return the count of each pattern type across all analysed heads."""
        counts = {}
        for layer_idx in range(len(self.attention_activations)):
            for head_idx in range(self.n_heads):
                result = self.analyse_head(layer_idx, head_idx)
                if result is not None:
                    counts[result.pattern_type] = counts.get(result.pattern_type, 0) + 1
        return counts
