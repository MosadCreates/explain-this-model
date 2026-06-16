import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    estimated_cost_usd: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    neuron_count: int
    head_count: int
    provider: str


# Approximate costs per 1K tokens (USD)
PROVIDER_COSTS = {
    "gemini": {
        "gemini-2.0-flash": {"input_per_1k": 0.0001, "output_per_1k": 0.0004},
    },
    "claude": {
        "claude-3-haiku-20240307": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
        "claude-3-sonnet-20240229": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    },
}

# Average tokens per neuron explanation
AVG_NEURON_INPUT_TOKENS = 300
AVG_NEURON_OUTPUT_TOKENS = 80
AVG_HEAD_INPUT_TOKENS = 250
AVG_HEAD_OUTPUT_TOKENS = 80
AVG_BATCH_OVERHEAD_TOKENS = 100


def estimate_api_cost(
    n_neurons: int,
    n_heads: int,
    prompt_length_tokens: int = 0,
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
    batch_size: int = 5,
) -> CostEstimate:
    """Estimate the cost of generating explanations via the API.

    Provides an estimate before any API calls are made, which is displayed
    in the UI so users know the expected cost before submitting.

    Args:
        n_neurons: Number of neuron explanations to generate.
        n_heads: Number of attention head explanations to generate.
        prompt_length_tokens: Length of the user's prompt (adds to input tokens).
        provider: Explanation provider ("gemini" or "claude").
        model: Specific model name.
        batch_size: Number of neurons per batch call.

    Returns:
        CostEstimate dataclass with estimated costs.
    """
    if provider == "claude" and model == "gemini-2.0-flash":
        model = "claude-3-haiku-20240307"
    costs = PROVIDER_COSTS.get(provider, {}).get(model, {"input_per_1k": 0.0, "output_per_1k": 0.0})

    n_batches = (n_neurons + batch_size - 1) // batch_size if n_neurons > 0 else 0
    n_single_calls = n_heads + n_batches

    input_tokens = (
        prompt_length_tokens
        + n_batches * (AVG_NEURON_INPUT_TOKENS * min(batch_size, n_neurons) + AVG_BATCH_OVERHEAD_TOKENS)
        + n_heads * AVG_HEAD_INPUT_TOKENS
    )
    output_tokens = (
        n_neurons * AVG_NEURON_OUTPUT_TOKENS
        + n_heads * AVG_HEAD_OUTPUT_TOKENS
    )

    estimated_cost = (
        (input_tokens / 1000) * costs["input_per_1k"]
        + (output_tokens / 1000) * costs["output_per_1k"]
    )

    return CostEstimate(
        estimated_cost_usd=round(estimated_cost, 6),
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        neuron_count=n_neurons,
        head_count=n_heads,
        provider=provider,
    )


class UsageTracker:
    """Tracks API usage per job for auditing and cost control.

    Records token usage and costs for each explanation generation request.
    In production, this data would be persisted to Redis or the database.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._daily_total: float = 0.0
        self._daily_tokens_in: int = 0
        self._daily_tokens_out: int = 0
        self._last_reset: float = time.time()

    def record_usage(
        self,
        job_id: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
    ) -> None:
        """Record API usage for a job."""
        if provider == "claude" and model == "gemini-2.0-flash":
            model = "claude-3-haiku-20240307"
        costs = PROVIDER_COSTS.get(provider, {}).get(model, {"input_per_1k": 0.0, "output_per_1k": 0.0})
        cost = (input_tokens / 1000) * costs["input_per_1k"] + (output_tokens / 1000) * costs["output_per_1k"]

        self._jobs[job_id] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "provider": provider,
            "timestamp": time.time(),
        }
        self._daily_total += cost
        self._daily_tokens_in += input_tokens
        self._daily_tokens_out += output_tokens
        logger.info("Recorded usage for job %s: %.6f USD (%d in + %d out)", job_id, cost, input_tokens, output_tokens)

    def get_job_usage(self, job_id: str) -> Optional[dict]:
        """Get usage data for a specific job."""
        return self._jobs.get(job_id)

    def get_daily_usage(self) -> dict:
        """Get cumulative daily usage stats."""
        now = time.time()
        if now - self._last_reset > 86400:
            self._reset_daily()

        return {
            "total_cost_usd": round(self._daily_total, 6),
            "total_input_tokens": self._daily_tokens_in,
            "total_output_tokens": self._daily_tokens_out,
            "job_count": len(self._jobs),
        }

    def check_daily_cap(self, cap_usd: float = 5.0) -> bool:
        """Check if the daily spend cap has been exceeded.

        Returns True if the cap is NOT exceeded (OK to make more requests).
        Returns False if the cap IS exceeded (reject new requests).
        """
        now = time.time()
        if now - self._last_reset > 86400:
            self._reset_daily()
        return self._daily_total < cap_usd

    def _reset_daily(self) -> None:
        self._daily_total = 0.0
        self._daily_tokens_in = 0
        self._daily_tokens_out = 0
        self._last_reset = time.time()
        logger.info("Daily usage counters reset")


class RateLimiter:
    """Sliding window rate limiter for API usage.

    Uses a simple in-memory counter. In production, this would use Redis
    for distributed rate limiting.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: list[float] = []

    def check(self) -> bool:
        """Check if a request is allowed under the rate limit.

        Returns True if allowed, False if rate limited.
        """
        now = time.time()
        cutoff = now - self.window_seconds
        self._requests = [t for t in self._requests if t > cutoff]

        if len(self._requests) >= self.max_requests:
            return False

        self._requests.append(now)
        return True

    def remaining(self) -> int:
        """Return the number of remaining requests in the current window."""
        now = time.time()
        cutoff = now - self.window_seconds
        self._requests = [t for t in self._requests if t > cutoff]
        return max(0, self.max_requests - len(self._requests))

    def reset(self) -> None:
        self._requests.clear()
