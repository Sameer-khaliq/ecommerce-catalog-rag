import time
from typing import Callable, TypeVar

from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)
T = TypeVar("T")


class CircuitBreaker:
    """Tracks consecutive failures; opens after a threshold so we stop hammering a
    dead API. A stateful counter, not business logic — the one class justified here."""

    def __init__(self, failure_threshold: int = 5):
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.is_open = False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            log_with_context(logger, "error", "circuit breaker opened", failures=self.consecutive_failures)


_breaker = CircuitBreaker()


def call_with_resilience(
    fn: Callable[[], T],
    product_id: str,
    stage: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T | dict:
    """Retries fn with exponential backoff. If the circuit is open, skips the call
    and returns a degradation stub instead of crashing the whole pipeline run."""
    if _breaker.is_open:
        log_with_context(logger, "warning", "circuit open, skipping call", product_id=product_id, stage=stage)
        return {"_degraded": True, "_reason": "circuit_open"}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            _breaker.record_success()
            return result
        except Exception as e:
            last_error = e
            _breaker.record_failure()
            log_with_context(
                logger, "warning", "call failed, retrying",
                product_id=product_id, stage=stage, attempt=attempt, error=str(e),
            )
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))

    log_with_context(
        logger, "error", "call failed after all retries, degrading gracefully",
        product_id=product_id, stage=stage, error=str(last_error),
    )
    return {"_degraded": True, "_reason": str(last_error)}