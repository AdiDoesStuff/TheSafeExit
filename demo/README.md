# TheSafeExit — Demo

Two ways to see the project run, both from the repository root:

| Command | What it does | Time |
|---|---|---|
| `demo\run_demo.bat` | Starts the web editor and opens the browser | a few seconds |
| `.venv\Scripts\python.exe demo\showcase.py` | Prints the headline numbers and saves four plots | under a minute |

Both are uncalibrated: they compare designs under stated assumptions and do not
predict real evacuation times.

For a click-by-click run sheet, including the expected numbers for each button,
see [`DEMO_CHECKLIST.md`](DEMO_CHECKLIST.md).

---

## 1. One click: `run_demo.bat`

Double-click `demo\run_demo.bat` (or run it from a terminal). It:

1. checks that `.venv\Scripts\python.exe` exists, and prints the setup commands
   if it does not;
2. offers to regenerate the experiment evidence with
   `run_all.py --quick` — **skipped by default**, so the demo starts at once
   (press `Y` within 5 seconds to run it);
3. starts the Flask editor in its own window and waits until it answers;
4. opens <http://127.0.0.1:5050>.

Close the "TheSafeExit Web Editor" window to stop the server. Re-running the
launcher reuses an editor that is already listening on port 5050.

If the browser does not open, go to <http://127.0.0.1:5050> manually. If port
5050 is already busy, the launcher reuses the server listening there; close that
editor window first if you want a fresh one.

### Manual start

```bat
.venv\Scripts\python.exe demo\web_editor\app.py
```

Both start the same server on `127.0.0.1:5050`; the launcher just waits for
it to answer and opens the browser.

### Requirements

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-web.txt
```

The web editor needs `Flask` and `Flask-CORS` (`requirements-web.txt`). The
headless test suite skips its editor tests when Flask is absent, so the core
project works without these two packages.

---

## 2. The web editor

The editor is the main demo surface: draw a floor plan, then run any of the five
analyses against it. The left panel holds the tools and the scenario options; the
results appear as cards and tables on the right, and the grid itself becomes a
heatmap when a heatmap mode is selected.

### Drawing

| Action | Control |
|---|---|
| Draw walls / floor / exits | Tool buttons, then click or drag on the grid |
| Change grid size | Rows / columns inputs, then **Apply** |
| Load an example layout | **Template** — cycles simple room, two exits, pillar, barrier |
| Clear everything | **Clear** |
| Monte Carlo from a cell | `Shift`-click a floor cell after **Analyze** |

Cells are 0 = wall, 1 = floor, 2 = exit. The grid is validated before every
analysis: a layout with no exit, an isolated floor cell, or a value other than
0/1/2 is rejected with a readable message.

### The five analyses

**Analyze** builds the absorbing Markov chain for the current grid and solves

```text
(I − Q) t = 1
```

with one sparse direct solve. It fills the heatmap with the expected time to
exit from every cell and reports:

- worst-case and mean expected time (in steps),
- transient and absorbing state counts,
- the solver method (`spsolve`), the relative residual `‖(I − Q)t − 1‖ / ‖1‖`,
  the solve time and the number of unknowns.

The residual is required to be below `1e-8`; a solve that misses it is reported
as an error instead of being drawn. On the 20×20 simple room this is 324
unknowns, `J* = 27` steps, residual exactly 0, under a millisecond.

Switch the chain selector to **uniform** to compare the random walk with the
rational shortest-path walk.

**Monte Carlo** runs `M` sampled evacuations from one start cell and compares the
sample mean with the exact value from the same chain. It reports the mean, the
standard error, the exact value, the signed difference, the relative error and
`|difference| / standard error`. For the rational chain the walk is
deterministic, so the sample must reproduce the exact value exactly; for the
uniform chain the two agree within a couple of standard errors. `M` is
adjustable.

**Optimize** solves exit placement on the current obstacle layout:

- **Exhaustive** enumerates every legal boundary position, scores each by the
  rational worst-case BFS distance, and returns the true minimum on that set.
  Guaranteed optimal for the stated objective.
- **BFGS** is the older heuristic search over a rounded interpolation, kept as a
  comparison baseline. The gap `BFGS − exhaustive` is reported, including when it
  is zero.

The exhaustive optimum is loaded back into the editor so the next analysis uses
it. Per-wall BFGS rows, iteration counts, candidate counts and each stage's time
are shown.

**Simulate Panic** runs the coupled simulation: crowd density feeds a wall-aware
panic diffusion PDE, panic mixes each transition row between rational and
uniform, and agents are sampled from the resulting chain. The scrubber steps
through the recorded heatmap snapshots. The summary reports total steps, agents
evacuated, T50 / T90 / T100, mean evacuation time, blocked moves, the largest
blocked-move cells and the run time.

The **Scenario** panel applies to this button and to **Compare**:

| Control | Effect | Default |
|---|---|---|
| Panic decay γ | Subtracts `γu` from the field each step | 0 (off) |
| Cell & exit capacity | One agent per cell and one per exit per step; refused moves are counted | on in the editor UI |
| Speed / reaction-time traits | Samples per-agent traits from the bivariate Gaussian; correlation ρ selectable | off |
| Trait correlation ρ | Correlation between speed and reaction time | +0.75 |

Every option has an off switch, and the defaults in the API preserve the
original behaviour (decay 0, no capacity, traits off).

**Compare panic on / off** runs a common-seed ensemble twice — once with panic
disabled and once with the current scenario — and shows T50, T90 and T100
medians with 95% bootstrap intervals, the paired difference, blocked totals and
the finished fraction. Same seeds on both arms, so the difference is not noise.

On the 20×20 simple room with 30 agents, decay 0.15 and capacity on, this gives
T90 **27 → 58 steps (Δ +31, +115%)** over 20 seeds, with blocked moves rising
from 2,144 to 3,922.

### API

All endpoints are JSON POSTs, except health and the page itself.

| Endpoint | Purpose |
|---|---|
| `GET /` | The editor page |
| `GET /api/health` | Liveness, version, port, allowed CORS origins |
| `POST /api/analyze` | Chain build + sparse expected-time solve + heatmap |
| `POST /api/monte_carlo` | Sampled absorption times against the exact value |
| `POST /api/optimize` | Exhaustive optimum and BFGS baseline side by side |
| `POST /api/simulate` | Coupled panic simulation with snapshots and summary |
| `POST /api/ensemble` | Common-seed ensemble with T50/T90/T100 and intervals |

Errors are always JSON `{"ok": false, "error": "..."}` with a 4xx or 5xx status;
the traceback is logged on the server, never returned.

Safety limits: analysis up to 80×80, the dynamic simulation up to 40×40, BFGS
up to 40×40 (exhaustive search still runs above that). CORS accepts only
`http://127.0.0.1:5050` and `http://localhost:5050` unless `DEMO_CORS_ORIGINS`
overrides it. `/api/simulate` and `/api/ensemble` additionally refuse requests
that did not originate on the local machine.

---

## 3. The showcase script

```bash
.venv\Scripts\python.exe demo\showcase.py
```

It writes **only** to `experiments/out/showcase/` (created if missing), so it
never touches `demo/figures/` or the canonical experiment outputs.

| Step | What it shows | Figure |
|---|---|---|
| 1 | One sparse solve on the L-shaped hallway, then solve time and CG iterations against system size | `showcase_solver.png` |
| 2 | Exhaustive exit search over every legal position, BFGS gap, ensemble T90 of the finalists | `showcase_exit.png` |
| 3 | Panic off vs on, 20 common seeds, 40 agents, capacity 1 | `showcase_panic.png` |
| 4 | Hallway density sweep at 10% / 30% / 50% | `showcase_density.png` |

Typical results on this host:

```text
sparse hallway solve     126 unknowns, residual 0, under a millisecond
BFGS gap                 0.00 steps on the 12 x 12 grid
panic effect on T90      +44.0 steps over 20 common seeds
density T90              82 / 186 / 330 steps at 10% / 30% / 50%
total wall clock         under a minute
```

The script exits 0 on success and non-zero with a printed traceback on failure.
It sets `MPLBACKEND=Agg` before importing Matplotlib, so it runs headless.

---

## 4. Where the full evidence lives

| Path | Contents |
|---|---|
| `docs/RESULTS.md` | Canonical numbers from the last full E1–E7 run |
| `docs/EXPERIMENTS.md` | What each experiment asks and how to read its outputs |
| `docs/SCOPE_AND_CLAIMS.md` | Claims ledger: what may and may not be said |
| `experiments/run_all.py` | Harness; `--quick` for a smoke run, `--only E1,E3` for a subset |
| `experiments/out/` | CSVs, PNGs, `results.csv`, `results.md`, `environment.json` (gitignored, regenerated) |
| `demo/figures/` | Earlier phase figures, kept as committed artefacts |

```bash
.venv\Scripts\python.exe experiments\run_all.py            # full suite, about 16 min
.venv\Scripts\python.exe experiments\run_all.py --quick    # smoke run, about 6 min
.venv\Scripts\python.exe experiments\run_all.py --dry-run  # environment + plan only
```

---

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Python virtual environment not found` | Create it: `python -m venv .venv` then install both requirements files |
| Browser shows a connection error | The editor window closed or never started; run `run_demo.bat` again and read that window |
| `Address already in use` | Something else owns port 5050. Close the old "TheSafeExit Web Editor" window, or start the backend manually with `set PORT=5051` and open `http://127.0.0.1:5051` |
| "Backend not found" toast in the page | The page was opened from a file or another port; load `http://127.0.0.1:5050` |
| Analysis error about residual | The grid produced an ill-conditioned chain (usually a nearly isolated pocket). Fix the walls and retry |
| Simulate rejects the grid | The dynamic simulation is capped at 40×40 and needs every floor cell to reach an exit |
| Compare seems slow | It runs 20 seeded simulations twice; a few seconds is normal |
| Blank figures from a direct script run | Set `MPLBACKEND=Agg` or use the launcher, which does it for you |

---

## 6. What this demo does not claim

- It does not predict real evacuation times and is not calibrated against drill
  data.
- There is no casualty model. "Blocked moves" means agents were refused entry to
  a full cell, not that anyone was hurt.
- Panic is a dimensionless qualitative index, not a physical quantity.
- During a coupled run, `N_t = (I − Q_t)^(−1)` is a frozen-chain diagnostic, not
  a forecast; forecasts come from ensemble runs.
- The exhaustive exit search is optimal over enumerated boundary positions for
  the rational BFS objective — not over every possible building design.
