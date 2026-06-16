import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.cache import get_explanation_cache
from src.config import get_config
from src.database import create_job, get_job, get_result
from src.explanations.cost import RateLimiter, UsageTracker
from src.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ConfigResponse,
    HealthResponse,
    JobStatusResponse,
    ModelSearchResponse,
    ModelSearchResult,
    ResultResponse,
)
from src.tasks.analysis_task import run_analysis_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])

_config = get_config()

_rate_limiter = RateLimiter(
    max_requests=_config.get("rate_limiting", {}).get("max_jobs_per_hour", 10),
    window_seconds=_config.get("rate_limiting", {}).get("window_seconds", 3600),
)
_usage_tracker = UsageTracker()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/analyze", response_model=AnalyzeResponse, status_code=201)
def submit_analysis(req: AnalyzeRequest):
    model_name = req.model_name.strip()
    prompt = req.prompt.strip()

    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    if len(prompt) > _config.get("analysis", {}).get("max_prompt_tokens", 512) * 4:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long (max {_config.get('analysis', {}).get('max_prompt_tokens', 512)} tokens)",
        )

    if not _rate_limiter.check():
        retry_after = _rate_limiter.window_seconds
        logger.warning("Rate limit exceeded for /analyze")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {_rate_limiter.max_requests} requests per {retry_after}s.",
        )

    if not _usage_tracker.check_daily_cap(cap_usd=5.0):
        logger.warning("Daily cost cap exceeded")
        raise HTTPException(
            status_code=429,
            detail="Daily API spend cap exceeded. Try again tomorrow or set a higher cap.",
        )

    job_id = create_job(model_name, prompt)

    cache = get_explanation_cache()
    run_analysis_task.delay(
        model_name=model_name,
        prompt=prompt,
        job_id=job_id,
        cache_backend="redis" if cache.using_redis else "memory",
    )

    logger.info("Submitted analysis job %s for model '%s'", job_id, model_name)

    return AnalyzeResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result_url = None
    if job.status == "completed" and job.result_id:
        result_url = f"/api/results/{job_id}"

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        model_name=job.model_name,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
        result_url=result_url,
    )


@router.get("/results/{job_id}", response_model=ResultResponse)
def get_analysis_results(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status == "pending":
        return ResultResponse(job_id=job_id, status="pending")
    if job.status == "running":
        return ResultResponse(job_id=job_id, status="running")
    if job.status == "failed":
        return ResultResponse(job_id=job_id, status="failed", error_message=job.error_message)
    if job.status == "completed" and job.result_id:
        result_data = get_result(job.result_id)
        if result_data:
            return ResultResponse(job_id=job_id, status="completed", result=result_data)

    return ResultResponse(job_id=job_id, status="failed", error_message="Result data not found")


@router.get("/models/search", response_model=ModelSearchResponse)
def search_models(query: str = "", limit: int = 10):
    if not query or len(query) < 2:
        return ModelSearchResponse(results=[])

    try:
        import requests
        url = f"https://huggingface.co/api/models?search={query}&sort=likes&direction=-1&limit={limit}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for item in data:
                results.append(ModelSearchResult(
                    model_id=item.get("modelId", item.get("id", "")),
                    architecture=item.get("pipeline_tag"),
                    likes=item.get("likes", 0),
                ))
            return ModelSearchResponse(results=results)
    except Exception as e:
        logger.warning("Failed to search HuggingFace models: %s", e)

    return ModelSearchResponse(results=[])


@router.get("/models/suggested")
def get_suggested_models():
    return [
        {"name": "gpt2", "description": "GPT-2 Small — 124M params, causal LM, fast", "tags": ["causal", "small"]},
        {"name": "distilbert-base-uncased", "description": "DistilBERT — 66M params, masked LM", "tags": ["bert", "small"]},
        {"name": "facebook/opt-350m", "description": "OPT-350M — 350M params, causal LM", "tags": ["causal", "medium"]},
        {"name": "EleutherAI/pythia-70m", "description": "Pythia-70M — 70M params, causal LM", "tags": ["causal", "small"]},
        {"name": "google/tinybert-4l-312d", "description": "TinyBERT — 14M params, fast", "tags": ["bert", "tiny"]},
        {"name": "albert-base-v2", "description": "ALBERT Base — 12M params (shared)", "tags": ["bert", "small"]},
        {"name": "roberta-base", "description": "RoBERTa Base — 125M params, masked LM", "tags": ["bert", "medium"]},
        {"name": "microsoft/DialoGPT-small", "description": "DialoGPT Small — 117M params, dialogue", "tags": ["causal", "small"]},
    ]


@router.get("/models/validate")
def validate_model(model_name: str):
    from src.models.registry import ModelRegistry
    from src.models.quirks import infer_quirks
    from transformers import AutoTokenizer

    registry = ModelRegistry(max_size=1, max_model_size_mb=1024)
    try:
        loaded = registry.load_model(model_name, device="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        quirks = infer_quirks(loaded.model, tokenizer, "hello world")
        params = loaded.parameter_count
        est_seconds = max(5, int(params / 10_000_000))
        return {
            "valid": True,
            "parameter_count": params,
            "architecture": loaded.architecture_type,
            "estimated_analysis_seconds": min(est_seconds, 120),
            "n_layers": quirks.n_layers,
            "n_heads": quirks.n_heads,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


@router.get("/jobs/{job_id}/neuron/{layer_index}/{neuron_index}")
def get_neuron_detail(job_id: str, layer_index: int, neuron_index: int):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    result_data = get_result(job.result_id)
    if not result_data:
        raise HTTPException(status_code=404, detail="Result not found")

    neurons = result_data.get("neuron_results", [])
    for n in neurons:
        if n.get("layer_index") == layer_index and n.get("neuron_index") == neuron_index:
            return n
    raise HTTPException(status_code=404, detail="Neuron not found")


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    from src.database import delete_job as db_delete_job
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    db_delete_job(job_id)
    return None


@router.get("/config", response_model=ConfigResponse)
def get_api_config():
    cfg = get_config()
    return ConfigResponse(
        analysis=cfg.get("analysis", {}),
        explanation=cfg.get("explanation", {}),
        rate_limiting=cfg.get("rate_limiting", {}),
        cache=cfg.get("cache", {}),
        model=cfg.get("model", {}),
    )
