import json
import os

import numpy as np
import pytest

from experiments import e1_exact_vs_mc, e2_solver_scaling, run_all
from experiments.config import ExperimentConfig
from experiments.e4_density_sweep import _narrowest_cell
from experiments.layouts import e4_layout


def test_e1_smoke_writes_rows_and_plot(tmp_path):
    cfg = ExperimentConfig.quick_config(
        out_dir=str(tmp_path),
        e1_m_values=(200,),
        e1_n_cells=2,
    )
    result = e1_exact_vs_mc.run(cfg, str(tmp_path))
    assert result["rows"]
    assert len(result["rows"]) == 3 * 2
    for path in result["outputs"]:
        assert os.path.exists(path), path
    with open(os.path.join(tmp_path, "e1_rows.csv"), encoding="utf-8") as f:
        header = f.readline()
    assert "exact_steps" in header and "mc_mean" in header


def test_e2_smoke_reports_low_residuals(tmp_path):
    cfg = ExperimentConfig.quick_config(out_dir=str(tmp_path), e2_sizes=(10, 20))
    result = e2_solver_scaling.run(cfg, str(tmp_path))
    sparse = [r for r in result["rows"] if r["method"] == "sparse_spsolve"]
    cg = [r for r in result["rows"] if r["method"] == "cg_uniform"]
    bi = [r for r in result["rows"] if r["method"] == "bicgstab_rational"]
    assert sparse and cg and bi
    for row in sparse + cg + bi:
        assert row["status"] == "ok", row
        assert row["residual"] < 1e-8
    assert np.isfinite(result["svd_diagnostic"]["condition_number"])
    assert result["svd_diagnostic"]["condition_number"] > 1.0
    assert result["svd_diagnostic"]["full_rank_relative_error"] < 1e-8
    for path in result["outputs"]:
        assert os.path.exists(path), path


def test_run_all_dry_run_writes_plan_and_environment(tmp_path):
    code = run_all.main(["--dry-run", "--quick", "--out", str(tmp_path)])
    assert code == 0
    for name in ("environment.json", "config.json", "plan.json"):
        assert os.path.exists(os.path.join(tmp_path, name)), name
    with open(os.path.join(tmp_path, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    assert set(plan["experiments"]) == {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}


def test_e4_narrowest_cell_is_the_one_cell_choke():
    grid = e4_layout()
    assert _narrowest_cell(grid) == (21, 23)
