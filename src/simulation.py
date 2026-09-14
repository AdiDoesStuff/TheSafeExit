"""
simulation.py — Engine Orchestration: Coupled Evacuation Simulation Loop

Couples Engine C (Panic diffusion PDE) and Engine A (Absorbing Markov chain)
using dynamic Q-row convex combination.

At each timestep:
  1. Compute discrete crowd density rho_t from active agent coordinates.
  2. Advance the panic PDE u_t = PanicField.step(rho_t).
  3. Form the transient mixing vector lambda_t = clip(beta * panic(r, c), 0, 1).
  4. Construct dynamic transition matrices (Q_t, R_t) = build_mixed_QR(...).
  5. Optionally snapshot N_t = (I - Q_t)^(-1) via sparse LU solver.
  6. Discretely sample each active agent's next cell from the categorical
     distribution defined by row s of [Q_t | R_t].
  7. Absorb agents reaching exits and track evacuation times.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Any, Sequence

from src.floor_plan import CELL_WALL, CELL_WALKABLE, CELL_EXIT, compute_exit_distance_field
from src.markov import build_state_map, build_uniform_QR, build_rational_QR, build_mixed_QR
from src.panic import PanicField, compute_agent_density
from src.solvers import fundamental_matrix_lu
from src.heatmap import fundamental_to_heatmap


class EvacuationSimulation:
    """
    Orchestrates the coupled agent evacuation simulation over a floor plan grid.

    Parameters:
        floor_plan: 2D numpy array with cell encodings (0=Wall, 1=Walkable, 2=Exit).
        n_agents: Total number of agents in the simulation.
        D: Diffusion coefficient for panic contagion.
        k: Panic source-term gain (S = k * rho).
        beta: Rationality degradation parameter (lambda = clip(beta * panic, 0, 1)).
              Default beta=0.4 corresponds to 1 / 2.5 (full panic at crush density rho >= 5).
        initial_positions: Optional (N, 2) array of initial (row, col) coordinates.
        seed: Optional random seed for reproducible stochastic sampling.
    """

    def __init__(
        self,
        floor_plan: np.ndarray,
        n_agents: int = 100,
        D: float = 0.2,
        k: float = 0.5,
        beta: float = 0.4,
        initial_positions: Optional[Sequence[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ):
        self.floor_plan = np.asarray(floor_plan, dtype=int)
        self.rows, self.cols = self.floor_plan.shape
        self.n_agents = n_agents
        self.D = D
        self.k = k
        self.beta = beta
        self.rng = np.random.default_rng(seed)

        # 1. Geometry and Geodesic Distance Field
        self.distance_field = compute_exit_distance_field(self.floor_plan)
        (
            self.state_map,
            self.index_to_state,
            self.n_transient,
            self.n_absorbing,
        ) = build_state_map(self.floor_plan)

        if self.n_transient == 0:
            raise ValueError("Floor plan has no walkable transient states.")
        if self.n_absorbing == 0:
            raise ValueError("Floor plan has no absorbing exit states.")

        # 2. Static Precomputed Baseline Transition Matrices
        self.Q_uniform, self.R_uniform = build_uniform_QR(
            self.floor_plan, self.state_map, self.n_transient, self.n_absorbing
        )
        self.Q_rational, self.R_rational = build_rational_QR(
            self.floor_plan, self.distance_field, self.state_map, self.n_transient, self.n_absorbing
        )

        # 3. Engine C: Panic Contagion Field
        self.panic_field = PanicField(self.floor_plan, D=self.D, k=self.k)

        # 4. Agent Initialization
        self.agent_states = np.zeros(self.n_agents, dtype=int)  # 0..n_transient-1, or -1 when absorbed
        self.evacuation_times: Dict[int, int] = {}
        self.time_step = 0

        if initial_positions is not None:
            if len(initial_positions) != self.n_agents:
                raise ValueError(
                    f"Length of initial_positions ({len(initial_positions)}) != n_agents ({self.n_agents})"
                )
            for idx, (r, c) in enumerate(initial_positions):
                if (r, c) not in self.state_map or self.state_map[(r, c)] >= self.n_transient:
                    raise ValueError(f"Position ({r, c}) is not a valid transient walkable cell.")
                self.agent_states[idx] = self.state_map[(r, c)]
        else:
            # Uniformly place agents across available transient cells
            self.agent_states = self.rng.choice(self.n_transient, size=self.n_agents)

        # Pre-extract transient state coordinates for fast panic vector extraction
        self._transient_coords = [self.index_to_state[i] for i in range(self.n_transient)]
        self._transient_rows = np.array([r for r, c in self._transient_coords], dtype=int)
        self._transient_cols = np.array([c for r, c in self._transient_coords], dtype=int)

    def get_active_agent_coords(self) -> np.ndarray:
        """Returns (M, 2) array of (row, col) coordinates for currently active (un-evacuated) agents."""
        active_states = self.agent_states[self.agent_states >= 0]
        if len(active_states) == 0:
            return np.empty((0, 2), dtype=int)
        coords = [self.index_to_state[s] for s in active_states]
        return np.array(coords, dtype=int)

    def step(self, record_snapshot: bool = False) -> Dict[str, Any]:
        """
        Advances the coupled simulation by one discrete timestep.

        Parameters:
            record_snapshot: If True, computes the reduced 2D bottleneck heatmap snapshot.

        Returns:
            step_summary: Dictionary containing step metrics and optional snapshot record.
        """
        active_indices = np.where(self.agent_states >= 0)[0]
        n_active = len(active_indices)

        if n_active == 0:
            return {
                "step": self.time_step,
                "active_agents": 0,
                "evacuated_agents": self.n_agents,
                "mean_panic": 0.0,
                "max_panic": 0.0,
                "mean_lambda": 0.0,
                "snapshot": None,
            }

        # 1. Calculate agent crowd density
        active_coords = [self.index_to_state[self.agent_states[i]] for i in active_indices]
        density = compute_agent_density(active_coords, (self.rows, self.cols))

        # 2. Advance Panic Diffusion PDE (Engine C)
        self.panic_field.step(density)

        # 3. Form transient state panic vector and mixing vector lambda_t
        panic_vec = self.panic_field.u[self._transient_rows, self._transient_cols]
        lam_t = np.clip(self.beta * panic_vec, 0.0, 1.0)

        # 4. Build dynamic transition matrices (Q_t, R_t)
        Q_t, R_t = build_mixed_QR(
            self.Q_rational, self.R_rational, self.Q_uniform, self.R_uniform, lam_t
        )

        # 5. Snapshot Bottleneck Heatmap (Engine D: solve N_t and immediately reduce to 2D)
        snapshot = None
        if record_snapshot:
            N_t = fundamental_matrix_lu(Q_t)
            heatmap_2d = fundamental_to_heatmap(
                N_t, self.floor_plan, self.state_map, self.n_transient
            )

            # Identify top-3 bottleneck cells (walkable cells with highest expected steps)
            valid_coords = np.argwhere(
                (~np.isnan(heatmap_2d)) & (self.floor_plan == CELL_WALKABLE)
            )
            top3_cells: List[Tuple[int, int, float]] = []
            if len(valid_coords) > 0:
                values = heatmap_2d[valid_coords[:, 0], valid_coords[:, 1]]
                top_k_indices = np.argsort(values)[::-1][:3]
                for k_idx in top_k_indices:
                    r = int(valid_coords[k_idx, 0])
                    c = int(valid_coords[k_idx, 1])
                    top3_cells.append((r, c, float(values[k_idx])))

            snapshot = {
                "t": self.time_step + 1,
                "heatmap_2d": heatmap_2d,
                "top3_cells": top3_cells,
                "mean_panic": float(np.mean(panic_vec)),
                "active_agents": int(n_active),
            }
            # Note: raw N_t matrix is immediately released / garbage collected

        # 6. Discretely sample each active agent's next cell
        for agent_idx in active_indices:
            current_state = self.agent_states[agent_idx]

            # Extract transition probabilities for current transient state
            # Concatenated [Q_t[s, :] | R_t[s, :]] of length (n_transient + n_absorbing)
            q_row = Q_t.getrow(current_state)
            r_row = R_t.getrow(current_state)

            cand_indices = []
            cand_probs = []

            # Non-zero transient destinations (0 .. n_transient - 1)
            for j, p in zip(q_row.indices, q_row.data):
                if p > 0:
                    cand_indices.append(j)
                    cand_probs.append(p)

            # Non-zero absorbing exit destinations (n_transient .. n_transient + n_absorbing - 1)
            for k, p in zip(r_row.indices, r_row.data):
                if p > 0:
                    cand_indices.append(self.n_transient + k)
                    cand_probs.append(p)

            # Normalize probabilities to avoid floating point sum drift
            cand_probs = np.array(cand_probs, dtype=np.float64)
            prob_sum = cand_probs.sum()
            if prob_sum > 0:
                cand_probs /= prob_sum
            else:
                cand_probs = np.ones(len(cand_indices)) / len(cand_indices)

            next_state = self.rng.choice(cand_indices, p=cand_probs)

            if next_state < self.n_transient:
                # Agent moves to transient cell
                self.agent_states[agent_idx] = next_state
            else:
                # Agent stepped into an exit (absorbing state)
                self.agent_states[agent_idx] = -1
                self.evacuation_times[agent_idx] = self.time_step + 1

        self.time_step += 1

        return {
            "step": self.time_step,
            "active_agents": int(np.sum(self.agent_states >= 0)),
            "evacuated_agents": int(np.sum(self.agent_states == -1)),
            "mean_panic": float(np.mean(panic_vec)),
            "max_panic": float(np.max(panic_vec)),
            "mean_lambda": float(np.mean(lam_t)),
            "snapshot": snapshot,
        }

    def run(
        self,
        max_steps: int = 600,
        snapshot_every: int = 20,
        max_snapshots: int = 30,
    ) -> Dict[str, Any]:
        """
        Runs the simulation loop until all agents evacuate or max_steps is reached.

        Parameters:
            max_steps: Maximum number of discrete simulation timesteps.
            snapshot_every: Interval in timesteps to record reduced bottleneck snapshots.
            max_snapshots: Hard ceiling on the total number of snapshots stored.

        Returns:
            trajectory: Dictionary with history log, evacuation statistics, and snapshot list.
        """
        history = []
        snapshots: List[Dict[str, Any]] = []
        agent_position_history: List[List[List[int]]] = []

        for _ in range(max_steps):
            if np.all(self.agent_states == -1):
                break

            should_snapshot = (
                snapshot_every > 0
                and ((self.time_step + 1) % snapshot_every == 0)
                and (len(snapshots) < max_snapshots)
            )

            summary = self.step(record_snapshot=should_snapshot)
            if should_snapshot and summary["snapshot"] is not None:
                snapshots.append(summary["snapshot"])

            # Exclude snapshot dict from per-step scalar history log
            step_record = {k: v for k, v in summary.items() if k != "snapshot"}
            history.append(step_record)

            # Record per-agent positions
            curr_positions = [
                list(self.index_to_state[s]) if s >= 0 else [-1, -1]
                for s in self.agent_states
            ]
            agent_position_history.append(curr_positions)

        evacuated_count = len(self.evacuation_times)
        evac_times = list(self.evacuation_times.values())
        mean_evac_time = float(np.mean(evac_times)) if evac_times else float("inf")

        return {
            "total_steps": self.time_step,
            "n_agents": self.n_agents,
            "evacuated_agents": evacuated_count,
            "evacuation_rate": evacuated_count / self.n_agents,
            "mean_evacuation_time": mean_evac_time,
            "history": history,
            "snapshots": snapshots,
            "agent_position_history": agent_position_history,
        }
