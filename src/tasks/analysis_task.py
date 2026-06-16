import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from celery import Task

from src.analysis.aggregator import build_analysis_result
from src.analysis.attention_analyser import AttentionAnalyser
from src.analysis.neuron_analyser import NeuronAnalyser
from src.cache import get_activation_cache, get_explanation_cache
from src.config import get_config
from src.database import (
    create_job,
    store_result,
    update_job_status,
)
from src.explanations.batch import BatchExplanationGenerator
from src.explanations.generator import ExplanationGenerator, NullExplanationGenerator
from src.models.hooks import ActivationHookManager
from src.models.registry import ModelRegistry
from src.models.quirks import infer_quirks
from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_model_registry: ModelRegistry = None
_last_registry_init: float = 0
_registry_ttl: float = 0

_config = get_config()


def _get_registry() -> ModelRegistry:
    global _model_registry, _last_registry_init, _registry_ttl
    now = time.time()
    if _model_registry is None or (now - _last_registry_init) > _registry_ttl:
        cache_size = _config.get("cache", {}).get("model_cache_size", 3)
        max_model_mb = _config.get("model", {}).get("max_size_mb", 1024)
        _model_registry = ModelRegistry(max_size=cache_size, max_model_size_mb=max_model_mb)
        _last_registry_init = now
        _registry_ttl = 3600
    return _model_registry


def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


@celery_app.task(bind=True, name="run_analysis")
def run_analysis_task(
    self: Task,
    model_name: str,
    prompt: str,
    job_id: str = None,
    cache_backend: str = "memory",
) -> dict:
    if job_id is None:
        job_id = create_job(model_name, prompt)

    logger.info("Analysis job %s started: model=%s, prompt=%r", job_id, model_name, prompt[:80])

    try:
        update_job_status(job_id, "running")

        activation_cache = get_activation_cache()
        cache_key = f"{model_name}:{_prompt_hash(prompt)}"

        cached_activations = activation_cache.get(model_name, prompt) if cache_backend != "memory" else None

        if cached_activations:
            logger.info("Activation cache HIT for %s", cache_key)
            mlp_raw = cached_activations.get("mlp_captures", [])
            attn_raw = cached_activations.get("attn_captures", [])
            native_raw = cached_activations.get("native_attentions")
            tokens = cached_activations.get("tokens", [])
            n_layers = cached_activations.get("n_layers", 0)
            n_heads = cached_activations.get("n_heads", 12)
            quirks_raw = cached_activations.get("quirks", {})
            mlp_captures = mlp_raw
            attn_captures = attn_raw
            native_attentions = native_raw
        else:
            logger.info("Activation cache MISS for %s, running forward pass", cache_key)

            registry = _get_registry()
            loaded = registry.load_model(model_name, device="cpu")

            tokenizer = loaded.tokenizer
            model = loaded.model

            tok_outputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512,
            )
            input_ids = tok_outputs["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

            quirks = infer_quirks(model, tokenizer, prompt)

            hook_manager = ActivationHookManager(
                model=model,
                mlp_patterns=_config.get("model", {}).get("mlp_patterns"),
                attention_patterns=_config.get("model", {}).get("attention_patterns"),
            )

            with torch.no_grad(), hook_manager:
                outputs = model(input_ids, output_attentions=True)
                mlp_captures = hook_manager.get_mlp_activations()
                attn_captures = hook_manager.get_attention_activations()
                native_attentions = outputs.attentions if hasattr(outputs, "attentions") else None

            n_layers = max(
                len(mlp_captures) if mlp_captures else 0,
                len(attn_captures) if attn_captures else 0,
                getattr(loaded.config, "num_hidden_layers", 0),
            )
            n_heads = getattr(loaded.config, "num_attention_heads", 12)

            if not mlp_captures:
                logger.warning("No MLP activations captured for model %s", model_name)

        analysis_start = time.time()

        neuron_analyser = NeuronAnalyser(
            mlp_activations=mlp_captures,
            tokens=tokens,
            context_window_size=_config.get("analysis", {}).get("context_window_size", 5),
        )

        top_k_neurons = _config.get("analysis", {}).get("top_k_neurons", 20)
        all_neuron_results = []

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(neuron_analyser.compute_all_neurons_for_layer, layer_idx): layer_idx
                for layer_idx in range(len(mlp_captures))
            }
            for future in as_completed(futures):
                try:
                    all_neuron_results.extend(future.result())
                except Exception as e:
                    logger.warning("Layer analysis failed: %s", e)

        ranked_neurons = sorted(all_neuron_results, key=lambda r: r.max_activation, reverse=True)[:top_k_neurons]
        layer_summaries = neuron_analyser.compute_layer_summaries()

        attn_analyser = AttentionAnalyser(
            attention_activations=attn_captures,
            tokens=tokens,
            native_attentions=native_attentions,
            n_heads=n_heads,
        )
        top_k_heads = _config.get("analysis", {}).get("top_k_heads", 10)
        ranked_heads = attn_analyser.rank_top_k(k=top_k_heads)

        analysis_duration = time.time() - analysis_start

        explanation_start = time.time()
        gemini_key = os.environ.get("GOOGLE_API_KEY")
        claude_key = os.environ.get("ANTHROPIC_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")

        explanation_provider = os.environ.get("EXPLANATION_PROVIDER") or _config.get("explanation", {}).get("provider", "gemini")
        exp_batch_size = _config.get("explanation", {}).get("batch_size", 5)

        if (explanation_provider == "gemini" and gemini_key) or (explanation_provider == "claude" and claude_key) or (explanation_provider == "groq" and groq_key):
            generator = ExplanationGenerator(
                provider=explanation_provider,
                gemini_api_key=gemini_key,
                claude_api_key=claude_key,
                groq_api_key=groq_key,
                gemini_model=_config.get("explanation", {}).get("gemini_model", "gemini-2.0-flash"),
                claude_model=_config.get("explanation", {}).get("claude_model", "claude-3-haiku-20240307"),
                groq_model=_config.get("explanation", {}).get("groq_model", "llama-3.1-8b-instant"),
            )
        else:
            generator = NullExplanationGenerator()

        exp_cache = get_explanation_cache() if cache_backend == "redis" else None

        batch_gen = BatchExplanationGenerator(
            generator=generator,
            cache=exp_cache,
            batch_size=exp_batch_size,
            enable_caching=True,
        )

        bundle = batch_gen.generate_explanations_batch(
            neuron_results=ranked_neurons,
            attention_results=ranked_heads,
            model_name=model_name,
            tokens=tokens,
            total_layers=n_layers,
        )

        explanation_duration = time.time() - explanation_start

        neuron_explanations_list = [
            {
                "layer_index": exp.layer_index,
                "neuron_index": exp.neuron_index,
                "hypothesis": exp.hypothesis,
                "confidence": exp.confidence,
                "pattern_type": exp.pattern_type,
                "cached": exp.cached,
            }
            for exp in bundle.neuron_explanations
        ]

        head_explanations_list = [
            {
                "layer_index": exp.layer_index,
                "head_index": exp.head_index,
                "hypothesis": exp.hypothesis,
                "confidence": exp.confidence,
                "pattern_type": exp.pattern_type,
                "cached": exp.cached,
            }
            for exp in bundle.head_explanations
        ]

        top_neuron_exp = bundle.neuron_explanations[0].hypothesis if bundle.neuron_explanations else ""

        analysis_result = build_analysis_result(
            model_name=model_name,
            prompt=prompt,
            tokens=tokens,
            architecture_type="gpt_style",
            parameter_count=0,
            n_layers=n_layers,
            n_heads=n_heads,
            neuron_results=ranked_neurons,
            attention_results=ranked_heads,
            layer_summaries=layer_summaries,
            analysis_duration=analysis_duration,
            top_neuron_explanation=top_neuron_exp,
        )

        result_dict = analysis_result.to_dict()
        result_dict["explanations"] = {
            "neurons": neuron_explanations_list,
            "heads": head_explanations_list,
            "total_api_calls": bundle.total_api_calls,
            "total_cached": bundle.total_cached,
            "explanation_duration_seconds": round(explanation_duration, 2),
        }

        store_result(job_id, result_dict)

        update_job_status(job_id, "completed")

        logger.info("Analysis job %s completed in %.1fs", job_id, analysis_duration + explanation_duration)

        return {"job_id": job_id, "status": "completed"}

    except Exception as e:
        logger.exception("Analysis job %s failed: %s", job_id, e)
        update_job_status(job_id, "failed", error_message=str(e))
        return {"job_id": job_id, "status": "failed", "error": str(e)}
