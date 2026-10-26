"""Run E1-E7 and write every CSV, PNG and configuration under experiments/out/.

Usage
-----
    python experiments/run_all.py                 # full suite
    python experiments/run_all.py --quick         # small smoke configuration
    python experiments/run_all.py --only E1,E3    # selected experiments
    python experiments/run_all.py --dry-run       # write env/config/plan only

The script always writes ``environment.json`` beside the results. A failed
experiment is recorded in ``results.csv``/``results.md`` and does not prevent the
remaining experiments from running; the process exits non-zero at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.common import (  # noqa: E402
    ensure_out_dir,
    progress,
    utc_now,
    write_csv,
    write_json,
)
from experiments.config import ExperimentConfig  # noqa: E402
from experiments.env_info import write_environment_info  # noqa: E402

EXPERIMENTS = {
    "E1": ("experiments.e1_exact_vs_mc", "run"),
    "E2": ("experiments.e2_solver_scaling", "run"),
    "E3": ("experiments.e3_panic_capacity", "run"),
    "E4": ("experiments.e4_density_sweep", "run"),
    "E5": ("experiments.e5_exit_search", "run"),
    "E6": ("experiments.e6_traits", "run"),
    "E7": ("experiments.e7_robustness", "run"),
}


def _load_spec(exp_id: str):
    module_name, _ = EXPERIMENTS[exp_id]
    module = __import__(module_name, fromlist=["SPEC"])
    return module.SPEC


def _run_one(exp_id: str, cfg: ExperimentConfig, out_dir: str) -> Dict[str, Any]:
    module_name, function_name = EXPERIMENTS[exp_id]
    spec = _load_spec(exp_id)
    progress(f"{exp_id}: {spec.title}")
    t0 = time.perf_counter()
    try:
        module = __import__(module_name, fromlist=[function_name])
        result = getattr(module, function_name)(cfg, out_dir)
        seconds = time.perf_counter() - t0
        result.setdefault("seconds", seconds)
        result["status"] = "ok" if result.get("passed", False) else "failed_criterion"
        return result
    except Exception as exc:
        seconds = time.perf_counter() - t0
        log_dir = os.path.join(out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{exp_id.lower()}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"{type(exc).__name__}: {exc}\n\n")
            f.write(traceback.format_exc())
        return {
            "id": exp_id,
            "spec": spec.to_dict(),
            "passed": False,
            "status": "error",
            "seconds": seconds,
            "key_finding": f"{type(exc).__name__}: {exc}",
            "outputs": [log_path],
        }


def _write_report(out_dir: str, results: List[Dict[str, Any]], cfg: ExperimentConfig) -> None:
    rows = []
    for result in results:
        spec = result.get("spec", {})
        rows.append(
            {
                "id": result.get("id", ""),
                "status": result.get("status", "error"),
                "passed": bool(result.get("passed", False)),
                "seconds": round(float(result.get("seconds", 0.0)), 3),
                "key_finding": result.get("key_finding", ""),
                "outputs": "; ".join(str(p) for p in result.get("outputs", [])),
            }
        )
    write_csv(os.path.join(out_dir, "results.csv"), rows)

    lines = [
        "# TheSafeExit experiment results",
        "",
        f"Generated: {utc_now()}",
        f"Configuration: {'quick' if cfg.quick else 'full'}  ",
        f"Seeds: {cfg.seeds[0]}..{cfg.seeds[-1]} (first n used per experiment)  ",
        f"Max steps: {cfg.max_steps}",
        "",
        "All outputs are uncalibrated and intended for comparing designs, not for",
        "predicting real evacuation times.",
        "",
        "| ID | Status | Seconds | Key finding |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        spec = result.get("spec", {})
        finding = str(result.get("key_finding", "")).replace("|", "/")
        lines.append(
            f"| {result.get('id', '')} | {result.get('status', 'error')} | "
            f"{float(result.get('seconds', 0.0)):.1f} | {finding} |"
        )

    lines.extend(["", "## Pre-registered questions", ""])
    for result in results:
        spec = result.get("spec", {})
        lines.extend(
            [
                f"### {spec.get('id', result.get('id', ''))} - {spec.get('title', '')}",
                "",
                f"**Question.** {spec.get('question', '')}",
                "",
                f"**Method.** {spec.get('method', '')}",
                "",
                f"**Expected.** {spec.get('expected', '')}",
                "",
                f"**Failure criterion.** {spec.get('failure_criterion', '')}",
                "",
                f"**Result.** {result.get('key_finding', '')}",
                "",
                "**Outputs.** " + ", ".join(f"`{p}`" for p in result.get("outputs", [])),
                "",
            ]
        )
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke configuration")
    parser.add_argument("--only", default="", help="comma-separated experiment ids, e.g. E1,E3")
    parser.add_argument("--out", default=os.path.join("experiments", "out"))
    parser.add_argument("--dry-run", action="store_true", help="write environment and plan only")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true", help="list experiment ids and exit")
    args = parser.parse_args(argv)

    if args.list:
        for exp_id, (module_name, _) in EXPERIMENTS.items():
            spec = _load_spec(exp_id)
            print(f"{exp_id}: {spec.title} ({module_name})")
        return 0

    out_dir = ensure_out_dir(args.out)
    cfg = ExperimentConfig.quick_config(out_dir=out_dir) if args.quick else ExperimentConfig(out_dir=out_dir)
    cfg.out_dir = out_dir

    env = write_environment_info(os.path.join(out_dir, "environment.json"))
    write_json(os.path.join(out_dir, "config.json"), cfg.to_dict())
    progress(f"environment: python {env.get('python_version', '').split()[0]}, "
             f"numpy {env.get('numpy_version')}, scipy {env.get('scipy_version')}, "
             f"commit {env.get('git_commit')}, dirty={env.get('git_dirty')}")

    requested = [part.strip().upper() for part in args.only.split(",") if part.strip()]
    if requested:
        unknown = [exp for exp in requested if exp not in EXPERIMENTS]
        if unknown:
            parser.error(f"unknown experiment id(s): {', '.join(unknown)}")
        selected = requested
    else:
        selected = list(EXPERIMENTS)

    if args.dry_run:
        plan = {
            "experiments": selected,
            "quick": cfg.quick,
            "config": cfg.to_dict(),
            "environment": env,
            "generated_utc": utc_now(),
        }
        write_json(os.path.join(out_dir, "plan.json"), plan)
        print("dry-run: environment, config and plan written to", out_dir)
        return 0

    results: List[Dict[str, Any]] = []
    for exp_id in selected:
        result = _run_one(exp_id, cfg, out_dir)
        results.append(result)
        progress(
            f"{exp_id}: {result['status']} in {result.get('seconds', 0.0):.1f}s — "
            f"{result.get('key_finding', '')}"
        )
        if args.fail_fast and result["status"] == "error":
            break

    _write_report(out_dir, results, cfg)
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    progress(f"wrote {n_ok}/{len(results)} passing experiments to {out_dir}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
