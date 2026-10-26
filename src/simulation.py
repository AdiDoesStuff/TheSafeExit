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
        panic_enabled: bool = True,
        panic_decay: float = 0.0,
        cell_capacity: Optional[int] = None,
        exit_capacity: Optional[int] = None,
        use_traits: bool = False,
        traits: Optional[Tuple] = None,
        cell_size_m: float = 0.5,
        step_s: float = 0.4,
        max_moves_per_step: int = 2,
    ):
        floor_plan_arr = np.asarray(floor_plan)
        if floor_plan_arr.ndim != 2:
            raise ValueError("floor_plan must be a 2D array")
        if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)) or n_agents < 0:
            raise ValueError("n_agents must be a non-negative integer")
        for name, val in (("D", D), ("k", k), ("beta", beta), ("panic_decay", panic_decay)):
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")

        self.floor_plan = np.asarray(floor_plan, dtype=int)
        self.rows, self.cols = self.floor_plan.shape
        self.n_agents = int(n_agents)
        self.D = D
        self.k = k
        self.beta = beta
        self.panic_enabled = panic_enabled
        self.panic_decay = panic_decay
        self.cell_capacity = cell_capacity
        self.exit_capacity = exit_capacity
        if cell_capacity is not None:
            if isinstance(cell_capacity, bool) or not isinstance(cell_capacity, (int, np.integer)) or cell_capacity < 0:
                raise ValueError("cell_capacity must be a non-negative integer or None")
            self.cell_capacity = int(cell_capacity)
        if exit_capacity is not None:
            if isinstance(exit_capacity, bool) or not isinstance(exit_capacity, (int, np.integer)) or exit_capacity < 0:
                raise ValueError("exit_capacity must be a non-negative integer or None")
            self.exit_capacity = int(exit_capacity)
        if not np.isfinite(cell_size_m) or cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        if not np.isfinite(step_s) or step_s <= 0:
            raise ValueError("step_s must be positive")
        if isinstance(max_moves_per_step, bool) or not isinstance(max_moves_per_step, (int, np.integer)) or max_moves_per_step < 1:
            raise ValueError("max_moves_per_step must be a positive integer")
        self.use_traits = bool(use_traits)
        self.traits = traits
        self.cell_size_m = float(cell_size_m)
        self.step_s = float(step_s)
        self.max_moves_per_step = int(max_moves_per_step)
        self.rng = np.random.default_rng(seed)
        # capacity rng created only when capacity enabled
        self._capacity_enabled = (self.cell_capacity is not None) or (self.exit_capacity is not None)
        self.capacity_rng = np.random.default_rng([seed, 1]) if self._capacity_enabled else None

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
        self.panic_field = PanicField(self.floor_plan, D=self.D, k=self.k, decay=self.panic_decay)

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
            if self.cell_capacity is not None:
                counts = np.zeros(self.n_transient, dtype=int)
                for s in self.agent_states:
                    counts[s] += 1
                if np.any(counts > self.cell_capacity):
                    raise ValueError("Initial placement exceeds cell_capacity.")
        else:
            if self.cell_capacity is None or self.cell_capacity >= self.n_agents:
                self.agent_states = self.rng.choice(self.n_transient, size=self.n_agents)
            else:
                slots = np.repeat(np.arange(self.n_transient), self.cell_capacity)
                if self.n_agents > len(slots):
                    raise ValueError("n_agents exceeds cell_capacity * n_transient.")
                chosen = self.rng.choice(len(slots), size=self.n_agents, replace=False)
                self.agent_states = slots[chosen].astype(int)

        # Trait initialization
        self.speeds = np.ones(self.n_agents, dtype=float)
        self.reactions = np.zeros(self.n_agents, dtype=float)
        self.delay_steps = np.zeros(self.n_agents, dtype=int)
        self.budget = np.zeros(self.n_agents, dtype=float)
        if self.use_traits:
            if self.traits is not None:
                sp_arr = np.asarray(self.traits[0], dtype=float).reshape(-1)
                rt_arr = np.asarray(self.traits[1], dtype=float).reshape(-1)
                if sp_arr.shape[0] != self.n_agents or rt_arr.shape[0] != self.n_agents:
                    raise ValueError("traits arrays must have length n_agents")
                self.speeds = sp_arr.copy()
                self.reactions = rt_arr.copy()
            else:
                from src.gaussian_sampler import sample_agents
                traits_rng = np.random.default_rng([seed, 2])
                sampled = sample_agents(n_agents=self.n_agents, rng=traits_rng)
                self.speeds = sampled[:, 0].astype(float)
                self.reactions = sampled[:, 1].astype(float)
            self.delay_steps = np.ceil(self.reactions / self.step_s).astype(int)
            self.delay_steps = np.where(self.delay_steps < 0, 0, self.delay_steps)
            self.budget = np.zeros(self.n_agents, dtype=float)

        # Pre-extract transient state coordinates for fast panic vector extraction
        self._transient_coords = [self.index_to_state[i] for i in range(self.n_transient)]
        self._transient_rows = np.array([r for r, c in self._transient_coords], dtype=int)
        self._transient_cols = np.array([c for r, c in self._transient_coords], dtype=int)

        self.blocked_counts = np.zeros((self.rows, self.cols), dtype=int)

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
                "blocked_this_step": np.zeros((self.rows, self.cols), dtype=int),
            }

        if self.panic_enabled:
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
        else:
            panic_vec = np.zeros(self.n_transient, dtype=float)
            lam_t = np.zeros(self.n_transient, dtype=float)
            Q_t, R_t = self.Q_rational, self.R_rational

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
                "panic_2d": self.panic_field.u.copy() if self.panic_field is not None else None,
                "top3_cells": top3_cells,
                "mean_panic": float(np.mean(panic_vec)),
                "active_agents": int(n_active),
            }
            # Note: raw N_t matrix is immediately released / garbage collected

        # Determine number of moves per active agent this step
        active_list = list(active_indices)
        if self.use_traits:
            # update budgets for agents not delayed
            for agent_idx in active_list:
                if self.time_step >= self.delay_steps[agent_idx]:
                    self.budget[agent_idx] += self.speeds[agent_idx] * self.step_s / self.cell_size_m
            moves_remaining = {}
            for agent_idx in active_list:
                if self.time_step < self.delay_steps[agent_idx]:
                    moves_remaining[agent_idx] = 0
                else:
                    mv = int(np.floor(self.budget[agent_idx]))
                    if mv < 0:
                        mv = 0
                    mv = min(mv, self.max_moves_per_step)
                    moves_remaining[agent_idx] = mv
                    self.budget[agent_idx] -= mv
        else:
            moves_remaining = {agent_idx: 1 for agent_idx in active_list}

        blocked_this_step = np.zeros((self.rows, self.cols), dtype=int)

        # Helper to sample intended destination for one agent from a given state
        def sample_intended(state):
            q_row = Q_t.getrow(state)
            r_row = R_t.getrow(state)
            cand_indices = []
            cand_probs = []
            for j, p in zip(q_row.indices, q_row.data):
                if p > 0:
                    cand_indices.append(j)
                    cand_probs.append(p)
            for k, p in zip(r_row.indices, r_row.data):
                if p > 0:
                    cand_indices.append(self.n_transient + k)
                    cand_probs.append(p)
            cand_probs = np.array(cand_probs, dtype=np.float64)
            prob_sum = cand_probs.sum()
            if prob_sum > 0:
                cand_probs /= prob_sum
            else:
                cand_probs = np.ones(len(cand_indices)) / len(cand_indices)
            return int(self.rng.choice(cand_indices, p=cand_probs))

        # Process up to max_moves rounds
        for _round in range(self.max_moves_per_step if self.use_traits else 1):
            # candidates with remaining moves
            round_candidates = [a for a in active_list if moves_remaining.get(a, 0) > 0 and self.agent_states[a] >= 0]
            if not round_candidates:
                break
            # sample intended for each candidate based on current cell
            intended_map = {}
            for agent_idx in round_candidates:
                st = self.agent_states[agent_idx]
                if st < 0:
                    continue
                intended_map[agent_idx] = sample_intended(st)

            if self._capacity_enabled:
                occupancy = np.zeros(self.n_transient, dtype=int)
                for agent_idx in active_list:
                    s = self.agent_states[agent_idx]
                    if s >= 0:
                        occupancy[s] += 1
                exit_counts = np.zeros(self.n_absorbing, dtype=int)
                perm = self.capacity_rng.permutation(len(round_candidates))
                for idx_p in perm:
                    agent_idx = round_candidates[idx_p]
                    current_state = self.agent_states[agent_idx]
                    if current_state < 0:
                        moves_remaining[agent_idx] = 0
                        continue
                    dest = intended_map[agent_idx]
                    if dest == current_state:
                        moves_remaining[agent_idx] -= 1
                        continue
                    if dest >= self.n_transient:
                        exit_idx = dest - self.n_transient
                        accept = True
                        if self.exit_capacity is not None:
                            if exit_counts[exit_idx] >= self.exit_capacity:
                                accept = False
                        if accept:
                            self.agent_states[agent_idx] = -1
                            self.evacuation_times[agent_idx] = self.time_step + 1
                            exit_counts[exit_idx] += 1
                            occupancy[current_state] = max(0, occupancy[current_state] - 1)
                        else:
                            exit_coord = self.index_to_state[dest]
                            blocked_this_step[exit_coord] += 1
                            self.blocked_counts[exit_coord] += 1
                    else:
                        accept = True
                        if self.cell_capacity is not None:
                            if occupancy[dest] >= self.cell_capacity:
                                accept = False
                        if accept:
                            occupancy[current_state] = max(0, occupancy[current_state] - 1)
                            occupancy[dest] += 1
                            self.agent_states[agent_idx] = dest
                        else:
                            blocked_this_step[self.index_to_state[dest]] += 1
                            self.blocked_counts[self.index_to_state[dest]] += 1
                    moves_remaining[agent_idx] -= 1
            else:
                for agent_idx in round_candidates:
                    current_state = self.agent_states[agent_idx]
                    if current_state < 0:
                        moves_remaining[agent_idx] = 0
                        continue
                    dest = intended_map[agent_idx]
                    if dest == current_state:
                        moves_remaining[agent_idx] -= 1
                        continue
                    if dest < self.n_transient:
                        self.agent_states[agent_idx] = dest
                    else:
                        self.agent_states[agent_idx] = -1
                        self.evacuation_times[agent_idx] = self.time_step + 1
                    moves_remaining[agent_idx] -= 1

        self.time_step += 1

        return {
            "step": self.time_step,
            "active_agents": int(np.sum(self.agent_states >= 0)),
            "evacuated_agents": int(np.sum(self.agent_states == -1)),
            "mean_panic": float(np.mean(panic_vec)),
            "max_panic": float(np.max(panic_vec)),
            "mean_lambda": float(np.mean(lam_t)),
            "snapshot": snapshot,
            "blocked_this_step": blocked_this_step,
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
            "blocked_counts": self.blocked_counts.copy(),
        }
