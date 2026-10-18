from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "log_windows.csv"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "logsentinel.joblib"
DEFAULT_REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
MODEL_VERSION = "0.1.0"
ANOMALY_RATE = 0.18

NORMAL_LABEL = "normal"
ROOT_CAUSES = [
    "oom_killed",
    "image_pull_failed",
    "network_timeout",
    "crash_loop",
    "disk_pressure",
]
TARGET_LABELS = [NORMAL_LABEL, *ROOT_CAUSES]
