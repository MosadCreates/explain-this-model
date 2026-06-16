from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., min_length=1, max_length=255, description="HuggingFace model ID, e.g. 'gpt2'")
    prompt: str = Field(..., min_length=1, max_length=4096, description="Input text to analyse")


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    status: str
    model_name: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_url: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ModelSearchResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    architecture: Optional[str] = None
    likes: Optional[int] = None


class ModelSearchResponse(BaseModel):
    results: list[ModelSearchResult]


class ConfigResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    analysis: dict[str, Any]
    explanation: dict[str, Any]
    rate_limiting: dict[str, Any]
    cache: dict[str, Any]
    model: dict[str, Any]
