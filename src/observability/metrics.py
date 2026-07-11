import time
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)

_stage_timings: dict[str, list[float]] = {}


@contextmanager
def track_stage(stage_name: str):
    """Usage: with track_stage('retrieval'): ... — records elapsed ms automatically."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _stage_timings.setdefault(stage_name, []).append(elapsed_ms)
        log_with_context(logger, "info", "stage timed", stage=stage_name, latency_ms=round(elapsed_ms, 2))


def compute_percentiles(stage_name: str) -> dict:
    timings = _stage_timings.get(stage_name, [])
    if not timings:
        return {"stage": stage_name, "count": 0}
    return {
        "stage": stage_name,
        "count": len(timings),
        "p50": round(float(np.percentile(timings, 50)), 2),
        "p95": round(float(np.percentile(timings, 95)), 2),
        "p99": round(float(np.percentile(timings, 99)), 2),
        "max": round(max(timings), 2),
    }


def report_all_stages(output_path: str | None = None) -> dict:
    report = {stage: compute_percentiles(stage) for stage in _stage_timings}
    log_with_context(logger, "info", "latency report generated", stages=list(report.keys()))
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report