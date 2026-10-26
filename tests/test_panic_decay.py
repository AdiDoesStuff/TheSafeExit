import numpy as np
import pytest

from src.panic import PanicField, CELL_WALL, CELL_WALKABLE


def test_decay_zero_identical():
    fp = np.ones((10, 10), dtype=int)
    fp[0, :] = CELL_WALL
    fp[-1, :] = CELL_WALL
    fp[:, 0] = CELL_WALL
    fp[:, -1] = CELL_WALL
    density = np.zeros((10, 10))
    density[3:7, 3:7] = 2.0

    f0 = PanicField(fp, D=0.2, k=0.5, decay=0.0)
    f1 = PanicField(fp, D=0.2, k=0.5)
    for _ in range(50):
        f0.step(density)
        f1.step(density)
    assert np.array_equal(f0.u, f1.u)


def test_decay_reduces_field():
    fp = np.ones((8, 8), dtype=int)
    fp[0, :] = CELL_WALL
    fp[-1, :] = CELL_WALL
    fp[:, 0] = CELL_WALL
    fp[:, -1] = CELL_WALL
    f = PanicField(fp, D=0.2, k=0.5, decay=0.1)
    f.u[:] = 5.0
    f.u[fp == CELL_WALL] = 0.0
    maxes = []
    for _ in range(200):
        f.step(np.zeros_like(fp, dtype=float))
        maxes.append(float(f.u.max()))
    assert all(maxes[i] >= maxes[i + 1] - 1e-12 for i in range(len(maxes) - 1))
    assert maxes[-1] < 1e-6


def test_steady_state_uniform():
    fp = np.ones((8, 8), dtype=int)
    fp[0, :] = CELL_WALL
    fp[-1, :] = CELL_WALL
    fp[:, 0] = CELL_WALL
    fp[:, -1] = CELL_WALL
    rho = 3.0
    density = np.full(fp.shape, rho, dtype=float)
    density[fp == CELL_WALL] = 0.0
    k = 0.5
    decay = 0.2
    f = PanicField(fp, D=0.2, k=k, decay=decay, dt=0.01)
    # ensure dt stable
    for _ in range(5000):
        f.step(density)
    interior = f.u[1:-1, 1:-1]
    target = k * rho / decay
    assert np.allclose(interior, target, rtol=1e-3)


def test_walls_zero_nonnegative():
    fp = np.ones((8, 8), dtype=int)
    fp[0, :] = CELL_WALL
    fp[-1, :] = CELL_WALL
    fp[:, 0] = CELL_WALL
    fp[:, -1] = CELL_WALL
    f = PanicField(fp, D=0.2, k=0.5, decay=0.1)
    density = np.zeros(fp.shape)
    density[2:6, 2:6] = 4.0
    for _ in range(20):
        f.step(density)
    assert np.all(f.u[fp == CELL_WALL] == 0.0)
    assert np.all(f.u >= 0)


def test_validators():
    fp = np.ones((6, 6), dtype=int)
    fp[0, :] = CELL_WALL
    with pytest.raises(ValueError):
        PanicField(fp, D=0.0, k=0.5)
    with pytest.raises(ValueError):
        PanicField(fp, D=0.2, k=0.5, dx=0.0)
    with pytest.raises(ValueError):
        PanicField(fp, D=0.2, k=0.5, dy=-1.0)
    with pytest.raises(ValueError):
        PanicField(fp, D=0.2, k=-1.0)
    with pytest.raises(ValueError):
        PanicField(fp, D=0.2, k=0.5, safety_factor=0.0)
    with pytest.raises(ValueError):
        PanicField(fp, D=0.2, k=0.5, safety_factor=1.1)
    with pytest.raises(ValueError):
        PanicField(fp, D=0.2, k=0.5, decay=-0.1)
    # decay*dt >= 1
    with pytest.raises(ValueError):
        PanicField(fp, D=0.2, k=0.5, decay=10.0, dt=0.5, safety_factor=0.9)
    # non-finite
    with pytest.raises(ValueError):
        PanicField(fp, D=float("nan"), k=0.5)
    # max_stable_dt
    with pytest.raises(ValueError):
        PanicField.max_stable_dt(0.0)
