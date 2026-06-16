# Concepts

## What is a Neuron in a Transformer?

In a transformer model, a **neuron** refers to a single unit in the hidden dimension of a **multi-layer perceptron (MLP) layer**. Here is the precise structure:

### MLP Layer Anatomy

Each transformer block typically contains an MLP sublayer with two linear transformations separated by a non-linear activation function:

```
MLP(x) = W_out · activation(W_in · x + b_in) + b_out
```

Where:
- `W_in` has shape `[d_model, d_mlp]` — projects from the residual stream into the MLP hidden space
- `activation` is typically ReLU, GELU, or SwiGLU (depending on architecture)
- `W_out` has shape `[d_mlp, d_model]` — projects back to the residual stream
- **A single neuron** is column `i` of `W_in` and row `i` of `W_out`

For a given input token, the MLP produces an intermediate activation vector `a` of length `d_mlp`. The value at position `i` — `a[i]` — is the **activation value of neuron i** for that token.

### What an Activation Value Tells You

If neuron `i` has activation value `a[i] = 12.3` on token "cat" and `a[i] = 0.1` on token "the", this suggests neuron `i` detects something about the token "cat" that is absent from "the". The neuron acts as a **feature detector**: it "fires" (produces a high activation) when its preferred pattern is present in the input, and remains silent otherwise.

A ReLU or GELU-activated neuron with value zero is **dead** for that input — its preferred feature was not present.

## What is an Attention Head?

An **attention head** is one of the parallel attention mechanisms within a multi-head attention layer. Each head independently computes attention patterns:

```
AttentionHead(Q, K, V) = softmax(Q @ K^T / sqrt(d_head)) @ V
```

Where:
- `Q` (query), `K` (key), `V` (value) are projections of the input, each shaped `[seq_len, d_head]`
- The **attention pattern** `softmax(Q @ K^T / sqrt(d_head))` has shape `[seq_len, seq_len]`
- Entry `[i, j]` is a probability: how much token i attends to token j

### What an Attention Pattern Tells You

If head `h` in layer `l` produces an attention pattern where:
- Most query positions attend to position 0 → the head is a **first-token** or **sink** head
- Most query positions attend to the immediately preceding token → a **previous-token** head
- Position i attends to position i-1 strongly only when token i-1 and some earlier token match → an **induction head**
- No clear structure, high entropy → a **diffuse** head
- The pattern varies significantly with content → a **content-based** head (the most interesting class)

## What Does "Explaining" a Neuron Mean?

### The Automated Interpretability Pipeline

"Explaining" a neuron in natural language follows the methodology pioneered by Anthropic's automated interpretability research:

1. **Find the top-activating contexts**: Run a model on a prompt and identify which tokens cause a specific neuron to fire most strongly. Collect the surrounding context (±5 tokens) for each of the top-5 activating positions.

2. **Present these to a language model**: Show the language model the neuron's activation values alongside the text contexts, with the activating token highlighted. The prompt asks: "What linguistic or semantic feature might this neuron be detecting?"

3. **Generate an explanation hypothesis**: The language model produces a natural-language description such as:
   > "This neuron detects tokens related to numerical quantities, particularly when they appear in the context of measurements or counts."

4. **Assign a confidence level**: Not all explanations are equally reliable. The model classifies its hypothesis as `high`, `medium`, or `low` confidence based on how clearly the activating contexts support a single interpretation.

### Why This Works

Neurons in well-trained models exhibit **feature-like behaviour**: individual neurons consistently activate for specific patterns (e.g., the token " but" after a negation, or tokens in the first position of a sentence). A sufficiently capable language model (Gemini 2.0 Flash, Claude 3 Haiku) can infer these patterns from a small number of examples — the same way a human interpretability researcher would.

### Why This Is Hard

- **Polysemanticity**: A single neuron can detect multiple unrelated features (the superposition hypothesis). The explanation can only describe the most salient one for the given prompt.
- **Context dependence**: A neuron's behaviour may change depending on the model's other activations. The explanation is always conditional on the specific input.
- **Speculative**: The language model is generating a hypothesis about an inscrutable system. The explanation may be wrong, incomplete, or misleading.

### What This Tool Does NOT Claim

The tool is explicit about its limitations:

| What it does | What it does NOT do |
|---|---|
| Surfaces which neurons/heads activated most strongly | Does not prove these are the most important components |
| Describes activation patterns in natural language | Does not guarantee the description is correct |
| Ranks components by activation magnitude | Does not perform causal intervention (ablation/patching) |
| Classifies attention pattern types statistically | Does not verify the classification with behavioural tests |
| Caches and reuses explanations for identical inputs | Does not claim explanations generalise to all inputs |

The tool communicates these limitations directly in the UI with a prominent disclaimer. Honesty about interpretability's limitations is a feature, not a weakness.

## Architecture-Specific Considerations

### GPT-Style Models (Causal LM, Decoder-Only)

```
GPT-2, Pythia, LLaMA, Mistral, TinyStories
├── Embedding
├── TransformerBlock × N
│   ├── LayerNorm
│   ├── Attention (causal mask)
│   └── MLP (typically GELU or SwiGLU)
└── LayerNorm + LM Head
```

- MLP activation dimension is typically `4 × d_model` (or `8/3 × d_model` for SwiGLU)
- Attention uses a causal mask (each token can only attend to earlier tokens)
- For **gated MLP** architectures (LLaMA, Mistral), the MLP has three weight matrices instead of two: `W_gate`, `W_up`, `W_down`. The activation to extract is `W_gate(x) ⊙ silu(W_up(x))` — the element-wise gated activation before `W_down`.

### BERT-Style Models (Masked LM, Encoder-Only)

```
BERT, DistilBERT, RoBERTa
├── Embedding
├── TransformerLayer × N
│   ├── Attention (bidirectional, no mask)
│   └── MLP (typically GELU)
└── Pooler + Classifier
```

- Attention is bidirectional (every token can attend to every other token)
- The hooking mechanism is identical — only the attention masking differs
- Tokenisation uses WordPiece (BERT) vs BPE (GPT-2) — the tool handles this transparently via the loaded tokenizer

### Encoder-Decoder Models (T5, BART)

- The tool hooks both the encoder and decoder transformer blocks
- Cross-attention layers (where the decoder attends to encoder outputs) are identified and analysed separately
- MLP patterns in T5 use `T5FF` (a DenseActivationDense block with ReLU or GeGLU)

## The Hook System

The core technical challenge is placing hooks on the correct modules without knowing the model's architecture in advance. The solution is **pattern matching on module class names**:

```python
MLP_PATTERNS = ["MLP", "FFN", "FeedForward", "mlp", "ffn", "DenseReluDense"]
ATTENTION_PATTERNS = ["Attention", "MultiHead", "attention", "attn", "SelfAttention"]
```

This handles GPT-2 (`MLP`), BERT (`BertSelfAttention`, `BertIntermediate`), T5 (`T5FF`, `T5Attention`), and any other architecture whose module naming follows conventions — which is almost all HuggingFace models.

### Hook Lifecycle

```
register_hooks() ──▶ forward_pass() ──▶ capture_activations() ──▶ cleanup_hooks()
                                                                    │
                                                               [context manager
                                                                guarantees cleanup]
```

Hooks are registered as **context managers** (`__enter__` / `__exit__`) so they are always removed after the forward pass, even if an exception occurs. This prevents memory leaks from stale hooks.

## Analysis Metrics

### For Neurons

| Metric | Meaning | Formula |
|---|---|---|
| Max activation | Peak firing strength on this prompt | `max(a[:, neuron])` |
| Mean activation | Average firing level | `mean(a[:, neuron])` |
| Fraction active | How often the neuron fires | `sum(a[:, neuron] > 0) / seq_len` |
| Z-score | How unusual this activation is | `(value - mean_baseline) / std_baseline` |

### For Attention Heads

| Metric | Meaning | Formula |
|---|---|---|
| Entropy | How diffuse the attention is | `-sum(P[i,:] * log(P[i,:]))` per row |
| Focus score | How concentrated the attention is | `1 - entropy / log(seq_len)` |
| Pattern type | Classification of attention behaviour | See classification rules above |
| Induction score | Likelihood of being an induction head | Strength of `[A][B]...[A][B]` pattern |

## Processing Flow Summary

```
User Input
  │
  ▼
Validate Model (size check, architecture detection)
  │
  ▼
Create Job (status=pending, enqueue Celery task)
  │
  ▼
Load Model (from cache or HuggingFace hub)
  │
  ▼
Extract Activations (register hooks → forward pass → capture)
  │
  ▼
Analyse Neurons (rank by max activation, compute statistics)
  │
  ▼
Analyse Attention (rank by focus score, classify patterns)
  │
  ▼
Generate Explanations (Gemini/Claude API calls with caching)
  │
  ▼
Save Results (serialise to JSON, store in SQLite)
  │
  ▼
Return to User (via polling API)
```
