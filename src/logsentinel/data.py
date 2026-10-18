from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANOMALY_RATE, DEFAULT_DATA_PATH, RANDOM_STATE, ROOT_CAUSES


NORMAL_TEMPLATES = (
    "INFO request completed path=/api/v1/{resource} status=200 latency_ms={latency}",
    "INFO cache hit key=cache:{token}",
    "INFO pod heartbeat node=node-{node} pod=api-{pod}",
    "INFO model inference batch_size={batch} device=cuda:0 latency_ms={latency}",
    "INFO health check passed component=api pod=api-{pod}",
    "INFO scheduler assigned pod=api-{pod} node=node-{node}",
    "INFO collected metrics cpu={cpu} memory={memory} disk={disk}",
    "WARN retry scheduled attempt={attempt} latency_ms={latency}",
)


def _format_normal_line(rng: np.random.Generator) -> str:
    template = str(rng.choice(NORMAL_TEMPLATES))
    return template.format(
        resource=str(rng.choice(["items", "inference", "search", "health"])),
        token=f"{rng.integers(1000, 9999):x}",
        node=int(rng.integers(1, 12)),
        pod=int(rng.integers(1, 30)),
        batch=int(rng.choice([1, 2, 4, 8, 16])),
        latency=float(rng.lognormal(5.0, 0.35)),
        cpu=round(float(rng.beta(2.0, 5.0)), 3),
        memory=round(float(rng.beta(2.0, 4.5)), 3),
        disk=round(float(rng.beta(1.8, 6.0)), 3),
        attempt=int(rng.integers(1, 4)),
    )


def _format_anomaly_line(rng: np.random.Generator, cause: str) -> str:
    if cause == "oom_killed":
        return (
            "ERROR container terminated reason=OOMKilled "
            f"memory_mb={int(rng.integers(7800, 16000))} pod=api-{int(rng.integers(1, 30))}"
        )
    if cause == "image_pull_failed":
        return (
            "ERROR failed to pull image registry=registry.local/model "
            f"tag=v{int(rng.integers(1, 9))}.{int(rng.integers(0, 12))} "
            f"pod=api-{int(rng.integers(1, 30))}"
        )
    if cause == "network_timeout":
        return (
            "ERROR upstream timeout host=10.0."
            f"{int(rng.integers(1, 255))}.{int(rng.integers(1, 255))} status=504 "
            f"retry={int(rng.integers(2, 8))} latency_ms={int(rng.integers(1800, 9000))}"
        )
    if cause == "crash_loop":
        return (
            "ERROR process exited code=137 "
            f"restart_count={int(rng.integers(3, 10))} pod=api-{int(rng.integers(1, 30))}"
        )
    return (
        "ERROR no space left on device path=/var/lib/containerd "
        f"disk_usage={round(float(rng.beta(8.0, 1.8)), 3)} pod=api-{int(rng.integers(1, 30))}"
    )


def generate_log_windows(
    n_samples: int = 6000,
    anomaly_rate: float = ANOMALY_RATE,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a reproducible synthetic benchmark for log-window triage.

    Each row represents a short log window for one pod. A small fraction of
    anomalies intentionally omit the explicit fault line so that the model must
    combine log semantics with resource and error metrics.
    """
    if not 0 < anomaly_rate < 1:
        raise ValueError("anomaly_rate must be between 0 and 1")
    if n_samples < 50:
        raise ValueError("n_samples must be at least 50")

    rng = np.random.default_rng(seed)
    n_anomalies = int(round(n_samples * anomaly_rate))
    anomaly_indices = set(rng.choice(n_samples, size=n_anomalies, replace=False).tolist())
    rows: list[dict[str, object]] = []

    for index in range(n_samples):
        is_anomaly = index in anomaly_indices
        cause = str(rng.choice(ROOT_CAUSES)) if is_anomaly else "normal"

        latency_ms = float(rng.lognormal(5.1, 0.42))
        retry_count = int(rng.poisson(0.35))
        restart_count = int(rng.poisson(0.08))
        cpu_usage = float(rng.beta(2.2, 5.0))
        memory_usage = float(rng.beta(2.0, 4.5))
        disk_usage = float(rng.beta(1.8, 6.0))
        error_count = int(rng.poisson(0.35))
        request_count = max(1, int(rng.poisson(120)))
        hour = int(rng.integers(0, 24))

        # Hard negatives: maintenance jobs and handled errors look suspicious but
        # remain healthy windows. This prevents keyword-only learning.
        hard_negative = (not is_anomaly) and rng.random() < 0.18
        if hard_negative:
            cpu_usage = float(rng.beta(5.0, 2.0))
            memory_usage = float(rng.beta(4.5, 2.0))
        slow_normal = (not is_anomaly) and rng.random() < 0.05
        if slow_normal:
            latency_ms = float(rng.lognormal(6.8, 0.50))
        restart_normal = (not is_anomaly) and rng.random() < 0.03
        if restart_normal:
            restart_count += 2
        disk_normal = (not is_anomaly) and rng.random() < 0.04
        if disk_normal:
            disk_usage = float(rng.beta(5.5, 2.0))

        if is_anomaly:
            if cause == "oom_killed":
                memory_usage = float(rng.beta(5.5, 2.0))
                restart_count += int(rng.integers(1, 3))
                error_count += int(rng.integers(1, 3))
            elif cause == "image_pull_failed":
                retry_count += int(rng.integers(2, 6))
                request_count = max(0, request_count - int(rng.integers(40, 100)))
                error_count += int(rng.integers(1, 3))
            elif cause == "network_timeout":
                latency_ms = float(rng.lognormal(6.8, 0.55))
                retry_count += int(rng.integers(1, 5))
                error_count += int(rng.integers(1, 4))
            elif cause == "crash_loop":
                restart_count += int(rng.integers(2, 6))
                error_count += int(rng.integers(1, 4))
                request_count = max(0, request_count - int(rng.integers(20, 70)))
            else:
                disk_usage = float(rng.beta(5.5, 2.0))
                error_count += int(rng.integers(1, 4))

        # Measurement noise keeps the classes from being perfectly separable.
        cpu_usage = float(np.clip(cpu_usage + rng.normal(0, 0.035), 0, 1))
        memory_usage = float(np.clip(memory_usage + rng.normal(0, 0.035), 0, 1))
        disk_usage = float(np.clip(disk_usage + rng.normal(0, 0.025), 0, 1))

        line_count = int(rng.integers(3, 7))
        lines = [_format_normal_line(rng) for _ in range(line_count)]
        if (not is_anomaly) and rng.random() < 0.07:
            lines.append("WARN retry scheduled attempt=1 latency_ms=320")
        if hard_negative:
            lines.append("INFO batch job completed batch_size=128 device=cuda:0 status=200")
            lines.append("INFO rolling update completed pod=api-7 node=node-3")
        if (not is_anomaly) and rng.random() < 0.06:
            lines.append("ERROR upstream timeout handled by fallback status=200")

        has_explicit_fault = is_anomaly and (rng.random() < 0.62)
        if has_explicit_fault:
            position = int(rng.integers(0, len(lines) + 1))
            lines.insert(position, _format_anomaly_line(rng, cause))

        message = " | ".join(lines)
        severity = "ERROR" if is_anomaly else ("WARN" if "WARN" in message else "INFO")
        rows.append(
            {
                "window_id": f"win-{index:06d}",
                "message": message,
                "severity": severity,
                "latency_ms": round(latency_ms, 3),
                "retry_count": retry_count,
                "restart_count": restart_count,
                "cpu_usage": round(cpu_usage, 4),
                "memory_usage": round(memory_usage, 4),
                "disk_usage": round(disk_usage, 4),
                "error_count_5m": error_count,
                "request_count": request_count,
                "hour": hour,
                "is_anomaly": int(is_anomaly),
                "root_cause": cause,
                "target": cause if is_anomaly else "normal",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic log windows")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--anomaly-rate", type=float, default=ANOMALY_RATE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    frame = generate_log_windows(
        n_samples=args.samples,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"Wrote {len(frame)} rows to {args.output} "
        f"(anomaly rate={frame['is_anomaly'].mean():.3f})"
    )


if __name__ == "__main__":
    main()
