"""Shared helpers for the E1-E7 experiment harness."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentSpec:
    """The pre-registered question, method and pass/fail rule for one run."""

    id: str
    title: str
    question: str
    method: str
    expected: str
    failure_criterion: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Convert numpy / path values into JSON-serialisable Python values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2)
    return path


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k, "")) for k in fieldnames})
    return path


def save_figure(fig, path: str, dpi: int = 200) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_experiment_config(
    out_dir: str,
    spec: ExperimentSpec,
    cfg: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "experiment": spec.to_dict(),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else jsonable(cfg),
        "generated_utc": utc_now(),
    }
    if extra:
        payload.update(jsonable(extra))
    return write_json(os.path.join(out_dir, f"{spec.id.lower()}_config.json"), payload)


def artefact(out_dir: str, exp_id: str, suffix: str) -> str:
    return os.path.join(out_dir, f"{exp_id.lower()}_{suffix}")


# --------------------------------------------------------------------------- #
# Statistics and plotting
# --------------------------------------------------------------------------- #
def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-b, with a safe fallback when scipy is unavailable."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("a and b must have the same length")
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau

        value = kendalltau(a, b).statistic
        return float(value) if value is not None else float("nan")
    except Exception:
        concordant = 0
        discordant = 0
        for i in range(a.size):
            for j in range(i + 1, a.size):
                da = a[i] - a[j]
                db = b[i] - b[j]
                if da * db > 0:
                    concordant += 1
                elif da * db < 0:
                    discordant += 1
        denom = concordant + discordant
        return float((concordant - discordant) / denom) if denom else float("nan")


def rank_values(values: Sequence[float]) -> List[int]:
    """Rank 1 = best (largest value); NaN ranks last."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(-np.nan_to_num(arr, nan=-np.inf), kind="stable")
    ranks = np.empty(arr.size, dtype=int)
    ranks[order] = np.arange(1, arr.size + 1)
    return [int(r) for r in ranks]


def find_knee(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """
    Crude knee locator: x at the largest absolute second difference of y.

    The density sweep is coarse and noisy, so this is reported as a descriptive
    number, never as a formal changepoint estimate.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(y_arr)
    if mask.sum() < 3:
        return None
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    second = np.abs(np.diff(y_arr, n=2))
    if second.size == 0:
        return None
    return float(x_arr[int(np.argmax(second)) + 1])


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def progress(message: str) -> None:
    print(f"[experiments] {message}", flush=True)
