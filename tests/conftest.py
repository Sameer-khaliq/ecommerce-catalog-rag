import pytest
import src.enrichment.resilience as resilience_module


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Resets the module-level circuit breaker before every test so failures in
    one test don't leak into and pollute the next."""
    resilience_module._breaker.consecutive_failures = 0
    resilience_module._breaker.is_open = False
    yield