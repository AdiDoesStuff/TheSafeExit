import os
import pytest
import numpy as np

from src.floor_plan import get_simple_room, CELL_WALKABLE, CELL_EXIT
from src.simulation import EvacuationSimulation
from src.heatmap_sequence import (
    global_colorscale,
    plot_heatmap_grid,
    animate_heatmap_sequence,
    track_bottleneck_migration,
)


def test_global_colorscale():
    h1 = np.array([[np.nan, np.nan], [10.0, 25.0]])
    h2 = np.array([[np.nan, 5.0], [30.0, 15.0]])

    snapshots = [
        {"t": 1, "heatmap_2d": h1},
        {"t": 2, "heatmap_2d": h2},
    ]

    vmin, vmax = global_colorscale(snapshots)
    assert vmin == 5.0
    assert vmax == 30.0

    # Empty list handling
    empty_vmin, empty_vmax = global_colorscale([])
    assert empty_vmin == 0.0
    assert empty_vmax == 1.0


def test_snapshot_cadence_and_cap():
    grid = get_simple_room(8, 8, exit_width=2)
    sim = EvacuationSimulation(grid, n_agents=15, seed=42)

    results = sim.run(max_steps=50, snapshot_every=5, max_snapshots=4)
    snapshots = results["snapshots"]

    assert len(snapshots) == 4
    for i, snap in enumerate(snapshots):
        assert snap["t"] == (i + 1) * 5
        assert "heatmap_2d" in snap
        assert "top3_cells" in snap
        assert "mean_panic" in snap
        assert "active_agents" in snap


def test_animate_minimal(tmp_path):
    grid = get_simple_room(6, 6, exit_width=2)
    sim = EvacuationSimulation(grid, n_agents=10, seed=7)
    results = sim.run(max_steps=15, snapshot_every=5, max_snapshots=3)
    snapshots = results["snapshots"]

    out_gif = str(tmp_path / "test_animation.gif")
    anim = animate_heatmap_sequence(
        snapshots=snapshots,
        floor_plan=grid,
        history=results["history"],
        save_path=out_gif,
        fps=2,
    )

    assert os.path.exists(out_gif)
    assert os.path.getsize(out_gif) > 0


def test_bottleneck_migration_length():
    grid = get_simple_room(8, 8, exit_width=2)
    sim = EvacuationSimulation(grid, n_agents=20, seed=99)
    results = sim.run(max_steps=30, snapshot_every=10, max_snapshots=3)
    snapshots = results["snapshots"]

    migration = track_bottleneck_migration(snapshots)
    assert len(migration) == len(snapshots)

    for record in migration:
        assert "t" in record
        assert "cell" in record
        assert "value" in record
        r, c = record["cell"]
        assert 0 <= r < grid.shape[0]
        assert 0 <= c < grid.shape[1]
        assert grid[r, c] == CELL_WALKABLE
        assert record["value"] >= 0.0


def test_plot_heatmap_grid(tmp_path):
    grid = get_simple_room(8, 8, exit_width=2)
    sim = EvacuationSimulation(grid, n_agents=15, seed=12)
    results = sim.run(max_steps=30, snapshot_every=10, max_snapshots=3)
    snapshots = results["snapshots"]

    out_png = str(tmp_path / "test_grid.png")
    fig, axes = plot_heatmap_grid(snapshots, grid, ncols=2, save_path=out_png)

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 0
