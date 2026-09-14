"""
panic.py — Engine C: Panic Contagion as a Diffusion PDE

Physical model
--------------
Panic density u(x, y, t) obeys a 2D diffusion equation with a source term:

    du/dt = D * Laplacian(u) + S(x, y, t)

where:
    D        diffusion coefficient (how fast panic spreads to calm neighbors)
    S(x,y,t) source term — in this project, S = k * rho(x,y,t), where rho is
             the local agent density per cell at the current timestep. This
             is a deliberate design choice: density is already computed every
             timestep by the simulation loop, so no separate "emergency seed"
             mechanism is needed — crowding itself is what generates panic.

Numerical scheme
-----------------
Explicit FTCS (Forward-Time, Centered-Space), but implemented as a
flux-conductance (finite-volume) scheme rather than a naive 5-point stencil:

    - Every interior edge between two grid cells has a "conductance" of 1 if
      BOTH cells are passable (walkable or exit), and 0 if either side is a
      wall.
    - Flux across an edge = conductance * (u_neighbor - u_self).
    - A cell's new value = old value + dt * D * (net flux in) + dt * S.

This gives correct no-flux (Neumann) boundaries at walls for free: panic
cannot leak through a wall, and cannot "diffuse" into a room that isn't
actually adjacent through walkable space. No special-casing of boundary
cells is needed, and the whole update is vectorized (no per-cell Python
loops).

Stability
---------
This explicit scheme is only stable if:

    D * dt * (1/dx^2 + 1/dy^2) <= 0.5

`PanicField` computes the largest safe dt for you (`max_stable_dt`) and will
raise if you supply a dt that violates this condition — this is meant to be
impossible to silently get wrong.

This module implements Engine C ONLY. It knows nothing about Q, agents, or
Markov states — coupling panic values back into a per-agent transition row
mutation is solvers.py's job (see the "how to wire this in" note at the
bottom of this file).
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple, Union

# Match floor_plan.py's cell encoding.
CELL_WALL = 0
CELL_WALKABLE = 1
CELL_EXIT = 2


class UnstableTimestepError(ValueError):
    """Raised when a requested dt violates the CFL stability condition."""


class PanicField:
    """
    A 2D panic diffusion field coupled to a floor plan.

    Parameters
    ----------
    floor_plan : np.ndarray, shape (rows, cols)
        Same array used by floor_plan.py / markov.py (0=wall, 1=walkable,
        2=exit).
    D : float
        Diffusion coefficient.
    k : float
        Source-term gain: S = k * rho (rho = agents per cell this timestep).
    dx, dy : float
        Grid spacing. Defaults to 1.0 (unit cells), which is almost always
        the right choice unless you're modeling a floor plan at real-world
        scale.
    dt : float, optional
        Timestep. If omitted, it's set automatically to a safe fraction of
        the CFL limit (see `safety_factor`). If provided, it is validated
        against the CFL condition and rejected if unstable.
    safety_factor : float
        Fraction of the theoretical CFL limit to actually use (default 0.9).
        Keep this below 1.0 — the CFL bound is an equality-case limit, and
        floating-point / boundary effects mean you want margin under it.
    """

    def __init__(
        self,
        floor_plan: np.ndarray,
        D: float,
        k: float = 1.0,
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float | None = None,
        safety_factor: float = 0.9,
    ):
        floor_plan = np.asarray(floor_plan)
        if floor_plan.ndim != 2:
            raise ValueError("floor_plan must be a 2D array")

        self.floor_plan = floor_plan
        self.rows, self.cols = floor_plan.shape
        self.D = float(D)
        self.k = float(k)
        self.dx = float(dx)
        self.dy = float(dy)

        self.wall_mask = floor_plan == CELL_WALL
        self.passable_mask = ~self.wall_mask  # walkable OR exit

        # Edge conductances: 1 where both adjacent cells are passable.
        # h_cond connects (i, j)-(i, j+1); shape (rows, cols - 1)
        # v_cond connects (i, j)-(i+1, j); shape (rows - 1, cols)
        self.h_cond = (self.passable_mask[:, :-1] & self.passable_mask[:, 1:]).astype(float)
        self.v_cond = (self.passable_mask[:-1, :] & self.passable_mask[1:, :]).astype(float)

        safe_dt = self.max_stable_dt(self.D, self.dx, self.dy, safety_factor)
        if dt is None:
            self.dt = safe_dt
        else:
            if dt > safe_dt:
                raise UnstableTimestepError(
                    f"dt={dt} violates the CFL stability condition for "
                    f"D={self.D}, dx={self.dx}, dy={self.dy}. "
                    f"Max stable dt (with safety_factor={safety_factor}) is "
                    f"{safe_dt:.6g}. Either lower dt or raise dx/dy/lower D."
                )
            self.dt = float(dt)

        self.u = np.zeros((self.rows, self.cols), dtype=float)

    # ------------------------------------------------------------------ #
    # Stability
    # ------------------------------------------------------------------ #
    @staticmethod
    def max_stable_dt(D: float, dx: float = 1.0, dy: float = 1.0, safety_factor: float = 0.9) -> float:
        """
        Largest dt satisfying the explicit FTCS stability condition:

            D * dt * (1/dx^2 + 1/dy^2) <= 0.5

        Returns safety_factor * (theoretical limit), so the default already
        includes margin.
        """
        if D <= 0:
            raise ValueError("Diffusion coefficient D must be positive.")
        limit = 0.5 / (D * (1.0 / dx**2 + 1.0 / dy**2))
        return safety_factor * limit

    def is_stable(self, dt: float | None = None) -> bool:
        """Check whether a given dt (or the field's current dt) is CFL-stable."""
        dt = self.dt if dt is None else dt
        return self.D * dt * (1.0 / self.dx**2 + 1.0 / self.dy**2) <= 0.5

    # ------------------------------------------------------------------ #
    # Core update
    # ------------------------------------------------------------------ #
    def _laplacian(self, u: np.ndarray) -> np.ndarray:
        """Vectorized, wall-aware discrete Laplacian via edge fluxes."""
        # Horizontal contribution: flux_h[i,j] = conductance * (u[i,j+1] - u[i,j])
        lap_h = np.zeros_like(u)
        flux_h = self.h_cond * (u[:, 1:] - u[:, :-1])  # shape (rows, cols-1)
        lap_h[:, :-1] += flux_h
        lap_h[:, 1:] -= flux_h
        lap_h /= self.dx**2

        # Vertical contribution: flux_v[i,j] = conductance * (u[i+1,j] - u[i,j])
        lap_v = np.zeros_like(u)
        flux_v = self.v_cond * (u[1:, :] - u[:-1, :])  # shape (rows-1, cols)
        lap_v[:-1, :] += flux_v
        lap_v[1:, :] -= flux_v
        lap_v /= self.dy**2

        return lap_h + lap_v

    def step(self, density: np.ndarray | None = None, dt: float | None = None) -> np.ndarray:
        """
        Advance the panic field by one timestep.

        Parameters
        ----------
        density : np.ndarray, shape (rows, cols), optional
            Agents-per-cell this timestep (rho). Source term S = k * rho.
            If omitted, no source is injected this step (pure diffusion of
            whatever panic already exists).
        dt : float, optional
            Override the field's default dt for this single step. Validated
            against CFL just like the constructor.

        Returns
        -------
        np.ndarray
            The updated panic field (also stored as self.u).
        """
        step_dt = self.dt if dt is None else dt
        if dt is not None and not self.is_stable(step_dt):
            raise UnstableTimestepError(
                f"step() called with dt={step_dt}, which violates CFL "
                f"stability for this field's D={self.D}, dx={self.dx}, dy={self.dy}."
            )

        lap = self._laplacian(self.u)

        source = np.zeros_like(self.u)
        if density is not None:
            density = np.asarray(density, dtype=float)
            if density.shape != self.u.shape:
                raise ValueError(
                    f"density shape {density.shape} does not match grid shape {self.u.shape}"
                )
            source = self.k * density

        u_new = self.u + step_dt * (self.D * lap + source)
        u_new = np.clip(u_new, 0.0, None)   # panic density can't go negative
        u_new[self.wall_mask] = 0.0         # walls never hold panic

        self.u = u_new
        return self.u

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def panic_at(self, row: int, col: int) -> float:
        """Current panic level at a single grid cell."""
        return float(self.u[row, col])

    def total_panic(self) -> float:
        """Sum of panic across the whole grid — useful for a conservation sanity check."""
        return float(self.u.sum())

    def reset(self) -> None:
        """Zero out the panic field (e.g. between independent Monte Carlo runs)."""
        self.u[:] = 0.0


# ---------------------------------------------------------------------- #
# Integration Glue Helper
# ---------------------------------------------------------------------- #
def compute_agent_density(
    agent_coords: Union[np.ndarray, Sequence[Tuple[int, int]]],
    shape: Tuple[int, int]
) -> np.ndarray:
    """
    Computes a 2D density grid (rho) of shape (rows, cols) from agent cell coordinates.

    Parameters
    ----------
    agent_coords : np.ndarray of shape (N, 2) or list of (row, col) tuples
        Coordinates of all active agents at the current timestep.
    shape : tuple of (rows, cols)
        The 2D floor plan dimensions.

    Returns
    -------
    np.ndarray, shape (rows, cols)
        Count of agents located in each grid cell as a float array.
    """
    density = np.zeros(shape, dtype=float)
    if len(agent_coords) == 0:
        return density

    coords = np.asarray(agent_coords, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"agent_coords must have shape (N, 2), got {coords.shape}")

    # Vectorized fast binning via 1D flattened index bincount
    rows, cols = shape
    flat_indices = coords[:, 0] * cols + coords[:, 1]
    
    # Filter bounds to protect against invalid agent coordinates
    valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < rows) & (coords[:, 1] >= 0) & (coords[:, 1] < cols)
    if not np.all(valid_mask):
        flat_indices = flat_indices[valid_mask]

    counts = np.bincount(flat_indices, minlength=rows * cols)
    return counts.reshape(shape).astype(float)


# ---------------------------------------------------------------------- #
# How to wire this into simulation / solvers
# ---------------------------------------------------------------------- #
#
#   panic_field = PanicField(floor_plan, D=0.2, k=0.5)
#
#   for t in range(max_steps):
#       density = compute_agent_density(agent_positions, floor_plan.shape)   # rho_t
#       panic_field.step(density)                                            # Engine C
#
#       for agent in agents:
#           p = panic_field.panic_at(agent.row, agent.col)
#           mutate_agent_Q_row(agent, base_Q_row, panic_level=p)             # Engine A hook
#
#       ... Markov step, absorption check, N_t solve, etc.
#
# Keeping panic_field.step() and the Q-row mutation as two separate calls
# (rather than fusing them) is what makes this module independently testable
# and independently swappable — you can unit-test PanicField's stability and
# conservation properties without needing a single agent or Q matrix to exist.


if __name__ == "__main__":
    # Minimal standalone demo / sanity check.
    # A small room with walls on all four sides and one exit cell.
    fp = np.ones((10, 10), dtype=int)
    fp[0, :] = CELL_WALL
    fp[-1, :] = CELL_WALL
    fp[:, 0] = CELL_WALL
    fp[:, -1] = CELL_WALL
    fp[5, 9] = CELL_EXIT  # an exit punched into the right wall

    field = PanicField(fp, D=0.2, k=0.5)
    print(f"Auto-selected stable dt: {field.dt:.5f}")
    print(f"Is current dt stable? {field.is_stable()}")

    # Simulate a crowd cluster near the center, no exit-leak of panic itself.
    density = np.zeros_like(fp, dtype=float)
    density[4:6, 4:6] = 5.0  # 5 agents packed into 4 central cells

    print(f"\nTotal panic at t=0: {field.total_panic():.4f}")
    for step_num in range(1, 51):
        field.step(density)
        if step_num in (1, 10, 25, 50):
            print(f"t={step_num:3d}  total panic = {field.total_panic():8.4f}  "
                  f"peak = {field.u.max():.4f}  "
                  f"panic at (5,9) exit cell = {field.panic_at(5, 9):.4f}")

    # Sanity check: with a constant source and closed walls, total panic
    # should keep climbing (bounded only by how long you run it), and no
    # panic should have crossed into the fully-walled boundary cells.
    assert (field.u[fp == CELL_WALL] == 0).all(), "Panic leaked into a wall cell!"
    print("\nWall-isolation check passed: no panic leaked into wall cells.")

    # Instability check: requesting an unsafe dt should raise, not silently
    # produce garbage.
    try:
        PanicField(fp, D=0.2, k=0.5, dt=field.max_stable_dt(0.2) / 0.9 * 2.0)
        print("ERROR: should have raised UnstableTimestepError")
    except UnstableTimestepError as e:
        print(f"\nCFL guard works as expected: {e}")
