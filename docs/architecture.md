# Architecture

## Overview

Explain This Model is an **Interpretability-as-a-Service** tool that accepts any HuggingFace transformer model and a prompt, then returns a ranked, visualised breakdown of which neurons and attention heads fired most strongly, with natural-language explanations of what each component likely detects.

The architecture follows a three-layer design: **Model Adapter**, **Analysis Engine**, and **Explanation Generator** — each with clearly separated concerns.

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP API (FastAPI)                    │
│  POST /analyze  │  GET /jobs/{id}/status  │  GET /results│
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   Celery Task Worker                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ 1. Model     │  │ 2. Analysis  │  │ 3. Explanation│ │
│  │    Adapter   │─▶│    Engine    │─▶│    Generator  │ │
│  └──────────────┘  └──────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Three-Layer Design

### Layer 1 — Model Adapter (`src/models/`)

A generic interface that works with **any HuggingFace model** without model-specific hardcoding. It:

- **Discovers** the model's architecture by introspecting its `named_modules()` tree
- **Classifies** it as `gpt_style`, `bert_style`, or `encoder_decoder` via `infer_architecture_type()`
- **Identifies** MLP and attention layers by matching class name patterns (`"MLP"`, `"Attention"`, etc.) instead of hardcoded paths — this is what makes it model-agnostic
- **Registers** PyTorch `forward_hooks` on every MLP and attention module to capture activations
- **Extracts** activations during a single forward pass without modifying the model

**Key challenge**: Models have wildly different internal structures. GPT-2 uses `TransformerBlock.mlp`, BERT uses `BertSelfAttention` and `BertIntermediate`, T5 uses `T5FF` and `T5Attention`. The pattern-matching approach on class names handles all of these with the same code.

```python
# Pseudocode for hook placement
for name, module in model.named_modules():
    class_name = type(module).__name__
    if any(p in class_name for p in config.mlp_patterns):
        module.register_forward_hook(mlp_hook(name))
    if any(p in class_name for p in config.attention_patterns):
        module.register_forward_hook(attn_hook(name))
```

### Layer 2 — Analysis Engine (`src/analysis/`)

Takes the raw activation tensors captured by the hooks and computes interpretable statistics. It is entirely model-agnostic — it works on the shapes and values of tensors without knowing what model produced them.

**NeuronAnalyser** processes MLP activations (shape: `[batch, seq_len, d_mlp]`):
- Per-neuron activation statistics (max, mean, fraction active, z-score)
- Top-K ranking across all layers by max activation value
- Context window extraction (±5 tokens around the most-activating position)
- Dead neuron detection (zero-activation neurons)
- Activation heatmap data for visualisation

**AttentionAnalyser** processes attention weights (shape: `[batch, n_heads, seq_len, seq_len]`):
- Per-head pattern entropy (low entropy = focused attention)
- Pattern type classification (diagonal, previous_token, first_token, diffuse, content_based)
- Top-K ranking by focus score (1 - normalised entropy)
- Induction head detection (the `[A][B]...[A][B]` pattern)

**Why this is hard**: MLP hidden dimensions range from 256 (TinyStories) to 16,384 (large models). The analyser must compute statistics efficiently across all dimensions without assuming a fixed size.

### Layer 3 — Explanation Generator (`src/explanations/`)

Takes the top-K most-activated neurons and attention heads and sends structured prompts to a language model (Groq by default, with Gemini and Claude as alternatives) to generate natural-language feature descriptions.

**Neuron explanation prompt structure**:
```
You are an AI interpretability researcher analysing...
Model: {model_name}
Layer: {layer_index}, Neuron: {neuron_index}
Most activating token: "{token}" in context "...{left}>>>{token}<<<{right}..."
Top-5 activating positions: {table}
→ Generate JSON: {"hypothesis", "confidence", "pattern_type"}
```

**Design decisions**:
- Context windows are shown because the model needs surrounding tokens to infer the linguistic pattern
- Structured JSON output is required for deterministic parsing and display
- Confidence fields communicate that interpretability is inherently uncertain
- Multi-neuron batching groups up to 5 neurons per API call to reduce latency and cost

**Fallback**: If no API key is configured, explanations show as `"unavailable"` — the core activation analysis still works.

## Job Queue Architecture

### Why synchronous HTTP doesn't work

Extracting activations from a 500M-parameter model on CPU takes 5–15 seconds. Generating explanations via API adds another 2–5 seconds. Blocking an HTTP response for 10–20 seconds is unacceptable for both UX (the browser would time out) and server throughput (workers would be tied up waiting).

### The polling pattern

```
Client                          Server
  │                                │
  │  POST /api/analyze             │
  │───────────────────────────────▶│  Create job record (status=pending)
  │  {"job_id": "uuid", ...}       │  Enqueue Celery task
  │◀───────────────────────────────│
  │                                │
  │  GET /api/jobs/{id}/status     │
  │───────────────────────────────▶│  Return current status + progress
  │  {"status": "running", ...}    │
  │◀───────────────────────────────│
  │           ... poll every 1.5s ...│
  │                                │
  │  GET /api/jobs/{id}/results    │
  │───────────────────────────────▶│  Return full analysis JSON
  │  {"neurons": [...], ...}       │
  │◀───────────────────────────────│
```

### Why Celery + Redis

- **Redis** provides the message broker (Celery task queue) and result backend
- **Redis** also serves as the cache layer for model activations and explanations
- **Celery** manages the worker pool — tasks run in separate processes, freeing the HTTP server to handle requests
- **Persistence**: job state is stored in SQLite via SQLAlchemy, surviving worker restarts
- **Progress reporting**: Celery's `update_state()` mechanism enables real-time progress bars in the UI

## Caching Strategy

Three levels of caching:

| Level | Cache Key | TTL | Location | Purpose |
|---|---|---|---|---|
| 1 — Model | `model:{name}` | LRU (max 3) | In-memory | Avoid re-downloading and deserialising weights |
| 2 — Activation | `act:{model}:{prompt_hash}` | 1 hour | Redis | Same model+prompt combo across UI tweaks |
| 3 — Explanation | `exp:{model}:{layer}:{neuron}:{ctx_hash}` | 24 hours | Redis | Identical activation patterns across users |

## Data Flow (End-to-End)

```
1. User submits model name + prompt via POST /api/analyze
2. FastAPI creates an AnalysisJob record in SQLite (status=pending)
3. Celery task picks up the job:
   a. Stage: loading_model → ModelRegistry.load_model()
      - Checks LRU cache → loads from HF hub if not cached
      - Enforces size limit → rejects models > 1GB
   b. Stage: extracting_activations → ActivationHookManager
      - Registers hooks → runs forward pass with torch.no_grad()
      - Captures MLP activations + attention weights + token strings
   c. Stage: analysing → NeuronAnalyser + AttentionAnalyser
      - Computes top-K rankings, statistics, pattern classifications
   d. Stage: generating_explanations → ExplanationGenerator
      - Sends structured prompts to Groq/Gemini/Claude API
      - Caches explanations by (model, layer, neuron, context_hash)
   e. Stage: complete → Serialises AnalysisResult to JSON
      - Saves to SQLite → marks job complete
4. Frontend polls GET /api/jobs/{id}/status every 1.5s
5. Frontend fetches GET /api/jobs/{id}/results when complete
6. Results are displayed with animated transitions
```

## Limitations (Explicit)

This tool does **not**:
- Guarantee that natural-language explanations are correct — they are hypotheses
- Perform causal interventions (activation patching, ablation studies) — it is purely correlational
- Claim to fully explain model behaviour — it surfaces and describes activation patterns
- Work with models larger than 1GB (configurable) — this is a CPU-based tool
- Store model weights server-side beyond the in-memory LRU cache

## Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| API Layer | FastAPI + Uvicorn | Async-first, auto-docs, Pydantic v2 |
| Task Queue | Celery + Redis | Production-grade, persistent, retries |
| Database | SQLite + SQLAlchemy | Single-machine, no external DB needed |
| Model Loading | HuggingFace transformers + PyTorch | Industry standard, model-agnostic |
| Acceleration | transformer_lens (optional) | Faster activations for supported architectures |
| Analysis | numpy, scipy, einops | Efficient tensor operations |
| Explanations | Groq (default) / Google Gemini / Anthropic Claude | Structured NL feature descriptions |
| Frontend | Next.js 14 + Tailwind + shadcn/ui | Modern, accessible, fast |
| Visualisation | Recharts + custom CSS grids | Bar charts, heatmaps, attention matrices |
| State | Zustand + SWR | Lightweight client state + polling |
| Containerisation | Docker + docker-compose | Single-command deployment |
