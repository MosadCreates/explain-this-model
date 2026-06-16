"""Prompt templates for generating natural-language explanations of neurons and attention heads.

Prompt Design Philosophy:
- Context windows are shown because the language model needs surrounding tokens
  to infer the linguistic pattern a neuron detects — a single token in isolation
  is rarely informative.
- Structured JSON output is required for deterministic parsing and rendering in
  the dashboard UI.
- Confidence fields are included because interpretability is inherently uncertain
  — the tool must communicate this honestly rather than overclaiming certainty.
- Batch prompts group up to 5 neurons into a single API call to reduce latency
  and cost, since the overhead of an API call often exceeds the compute cost of
  generating additional explanations within a single response.
"""

import json
from typing import Any


SYSTEM_PROMPT = """You are an AI interpretability researcher analysing the internal activations of a transformer language model. Your task is to generate concise, specific hypotheses about what linguistic or semantic features individual neurons and attention heads detect.

Guidelines:
1. Be specific about the pattern you observe, referencing actual tokens.
2. If the pattern is unclear or the activations seem noisy, say so honestly.
3. Do not overclaim certainty — use the confidence field appropriately.
4. Focus on what the neuron/head DOES (what pattern it detects), not what it IS.
5. Keep each hypothesis to 1-2 sentences.
6. Respond ONLY with valid JSON, no additional text."""


def build_neuron_explanation_prompt(
    layer_index: int,
    neuron_index: int,
    total_layers: int,
    model_name: str,
    activating_token: str,
    activation_value: float,
    context_window_tokens: list[str],
    context_window_positions: list[int],
    top_activating_table: list[dict[str, Any]],
    activating_token_position: int,
) -> list[dict[str, str]]:
    """Build a structured prompt for explaining a single neuron.

    The prompt shows:
    1. Which model and layer the neuron belongs to
    2. The token and context that triggered the highest activation
    3. A table of the top-5 activating positions with their contexts
    4. A request for a structured JSON response

    This mirrors Anthropic's automated interpretability pipeline design.
    """
    left_context = _format_context(context_window_tokens, context_window_positions, activating_token_position, "left")
    right_context = _format_context(context_window_tokens, context_window_positions, activating_token_position, "right")

    rows = []
    for item in top_activating_table:
        rows.append(
            f"  Position {item['position']}: "
            f"token \"{item['token']}\" in context "
            f"\"{item['left_context']} >>>{item['token']}<<< {item['right_context']}\" "
            f"(activation: {item['activation']:.3f})"
        )

    prompt = (
        f"Model: {model_name}\n"
        f"Layer: {layer_index} of {total_layers}\n"
        f"Neuron index: {neuron_index}\n\n"
        f"This neuron activated most strongly on the following token in context:\n"
        f"  Token: \"{activating_token}\"\n"
        f"  Context: \"{left_context}>>>{activating_token}<<<{right_context}\"\n"
        f"  Activation value: {activation_value:.3f}\n\n"
        f"Here are the top activating positions where this neuron fired most strongly:\n"
        + "\n".join(rows) + "\n\n"
        f"Based on these activation patterns, write a concise 1-2 sentence hypothesis "
        f"about what linguistic or semantic feature this neuron might be detecting. "
        f"Be specific. If the pattern is unclear, say so honestly.\n\n"
        f"Respond ONLY with JSON: "
        f'{{"hypothesis": "...", "confidence": "high|medium|low", "pattern_type": "syntactic|semantic|positional|unclear"}}'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_attention_explanation_prompt(
    layer_index: int,
    head_index: int,
    model_name: str,
    pattern_type: str,
    focus_score: float,
    entropy: float,
    is_induction_head: bool,
    top_attended_pairs: list[dict[str, Any]],
    total_layers: int,
) -> list[dict[str, str]]:
    """Build a structured prompt for explaining an attention head."""
    pairs_str = "\n".join(
        f"  Query \"{p['query_token']}\" (pos {p['query_position']}) → "
        f"Key \"{p['key_token']}\" (pos {p['key_position']}), weight: {p['weight']:.3f}"
        for p in top_attended_pairs[:5]
    )

    induction_note = " This head shows strong evidence of being an induction head (attending from a token to the previous occurrence of the same token's successor)." if is_induction_head else ""

    prompt = (
        f"Model: {model_name}\n"
        f"Layer: {layer_index} of {total_layers}\n"
        f"Attention Head: {head_index}\n\n"
        f"Pattern classification: {pattern_type}\n"
        f"Focus score: {focus_score:.3f} (1.0 = perfectly focused, 0.0 = perfectly diffuse)\n"
        f"Entropy: {entropy:.3f}{induction_note}\n\n"
        f"Top attended token pairs:\n{pairs_str}\n\n"
        f"Based on this attention pattern, write a concise 1-2 sentence hypothesis "
        f"about what kind of information this attention head is routing between tokens. "
        f"Be specific about the pattern (e.g., 'this head attends from adjectives to the "
        f"nouns they modify').{induction_note}\n\n"
        f"Respond ONLY with JSON: "
        f'{{"hypothesis": "...", "confidence": "high|medium|low", "pattern_type": "{pattern_type}"}}'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_multi_neuron_prompt(
    neurons: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, str]]:
    """Build a batch prompt that asks the API to explain up to 5 neurons at once.

    Batching reduces API calls and latency. Each neuron is presented with
    its top-activating token, context, and activation table. The API is asked
    to return a JSON array of explanations, one per neuron.

    Expected response format:
    [
      {"neuron_index": 5, "hypothesis": "...", "confidence": "high", "pattern_type": "syntactic"},
      {"neuron_index": 12, "hypothesis": "...", "confidence": "medium", "pattern_type": "semantic"},
      ...
    ]
    """
    sections = []
    for n in neurons:
        left_ctx = _format_context(
            n.get("context_window", []), n.get("context_window_positions", []),
            n.get("activating_token_position", 0), "left"
        )
        right_ctx = _format_context(
            n.get("context_window", []), n.get("context_window_positions", []),
            n.get("activating_token_position", 0), "right"
        )

        rows = []
        for item in n.get("top_activating_table", []):
            rows.append(
                f"  Pos {item['position']}: \"{item['left_context']} >>>{item['token']}<<< {item['right_context']}\" "
                f"({item['activation']:.3f})"
            )

        section = (
            f"--- Neuron {n['neuron_index']} (Layer {n['layer_index']}) ---\n"
            f"Top token: \"{n['activating_token']}\"\n"
            f"Context: \"{left_ctx}>>>{n['activating_token']}<<<{right_ctx}\"\n"
            f"Max activation: {n['max_activation']:.3f}\n"
            f"Top activations:\n" + "\n".join(rows)
        )
        sections.append(section)

    prompt = (
        f"Model: {model_name}\n\n"
        f"Below are {len(neurons)} neurons from this model. For each neuron, "
        f"generate a 1-2 sentence hypothesis about what linguistic or semantic "
        f"feature it detects.\n\n"
        + "\n\n".join(sections) + "\n\n"
        f"Respond ONLY with a JSON array, one object per neuron in the same order:\n"
        f'[{{"neuron_index": ..., "hypothesis": "...", "confidence": "high|medium|low", '
        f'"pattern_type": "syntactic|semantic|positional|unclear"}}]'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _format_context(
    tokens: list[str],
    positions: list[int],
    center_position: int,
    side: str,
) -> str:
    """Format the context window tokens on one side of the center position."""
    if not tokens or not positions:
        return ""
    try:
        center_idx = positions.index(center_position) if center_position in positions else len(positions) // 2
    except ValueError:
        return " ".join(tokens)

    if side == "left":
        context = tokens[:center_idx]
    else:
        context = tokens[center_idx + 1:]

    text = "".join(context).replace("\n", " ").strip()
    if not text:
        return " "
    if side == "left" and not text.endswith(" "):
        text += " "
    if side == "right" and not text.startswith(" "):
        text = " " + text
    return text


def parse_explanation_response(response_text: str) -> list[dict[str, Any]]:
    """Parse the JSON response from the API, with graceful fallback for malformed JSON.

    Returns a list of explanation dicts with keys: hypothesis, confidence, pattern_type.
    If parsing fails entirely, returns a single fallback explanation.
    """
    text = response_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}")
        array_start = text.find("[")
        array_end = text.rfind("]")

        if array_start >= 0 and array_end > array_start:
            try:
                data = json.loads(text[array_start:array_end + 1])
            except json.JSONDecodeError:
                data = None
        elif json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(text[json_start:json_end + 1])
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if data is None:
        return [{"hypothesis": response_text[:200], "confidence": "low", "pattern_type": "unclear"}]

    if isinstance(data, dict):
        if "hypothesis" in data:
            return [data]
        return [{"hypothesis": str(data)[:200], "confidence": "low", "pattern_type": "unclear"}]

    if isinstance(data, list):
        valid = []
        for item in data:
            if isinstance(item, dict) and "hypothesis" in item:
                valid.append({
                    "hypothesis": item["hypothesis"],
                    "confidence": item.get("confidence", "low"),
                    "pattern_type": item.get("pattern_type", "unclear"),
                })
            elif isinstance(item, dict):
                valid.append({
                    "hypothesis": str(item.get("hypothesis", str(item)))[:200],
                    "confidence": item.get("confidence", "low"),
                    "pattern_type": item.get("pattern_type", "unclear"),
                })
        if valid:
            return valid

    return [{"hypothesis": "Explanation unavailable", "confidence": "low", "pattern_type": "unclear"}]


def build_neuron_context_table(
    tokens: list[str],
    activation_values: list[float],
    layer_index: int,
    neuron_index: int,
    context_window_size: int = 5,
) -> tuple[str, list[dict[str, Any]], str, int, float, list[str], list[int], int]:
    """Build the context table for a neuron's top activating positions.

    Returns:
        (activating_token, top_activating_table, center_left_context,
         activating_token_position, max_activation, context_window_tokens,
         context_window_positions, seq_len)
    """
    import numpy as np

    seq_len = len(tokens)
    acts = np.array(activation_values)

    max_pos = int(np.argmax(acts))
    max_val = float(acts[max_pos])
    activating_token = tokens[max_pos] if max_pos < len(tokens) else "[UNK]"

    half = context_window_size
    start = max(0, max_pos - half)
    end = min(seq_len, max_pos + half + 1)
    context_tokens = tokens[start:end]
    context_positions = list(range(start, end))

    top_indices = np.argsort(acts)[::-1][:5]
    top_table = []
    for pos in top_indices:
        pos = int(pos)
        l_start = max(0, pos - half)
        l_end = min(seq_len, pos + half + 1)
        left_ctx = "".join(tokens[l_start:pos])
        right_ctx = "".join(tokens[pos + 1:l_end])
        top_table.append({
            "position": pos,
            "token": tokens[pos] if pos < len(tokens) else "[UNK]",
            "activation": float(acts[pos]),
            "left_context": left_ctx.replace("\n", " ").strip(),
            "right_context": right_ctx.replace("\n", " ").strip(),
        })

    left_ctx = "".join(tokens[start:max_pos])
    right_ctx = "".join(tokens[max_pos + 1:end])

    return (
        activating_token,
        top_table,
        left_ctx.replace("\n", " ").strip(),
        max_pos,
        max_val,
        context_tokens,
        context_positions,
        seq_len,
    )
