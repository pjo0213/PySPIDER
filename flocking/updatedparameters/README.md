# Particle-scaled flocking with 10,000 particles

[01_updated_flocking.ipynb](01_updated_flocking.ipynb) generates 10,000 particles using the original box scaling and random sequential initializer in [particles.py](../particles.py). Continuous coarse-grained fields are evaluated on uniform spatial grids, with composite trapezoidal integration in space and saved time. A second independent analysis checks the matrices, column scales, fitted coefficients, ordered libraries and integration domains for exact equality.

The workflow retains the 12-unit duration and regression settings of [the existing dynamics notebook](../dynamics/01_flocking_discovery.ipynb). All changes and generated artifacts are contained in this directory; the original simulator, dynamics notebooks and previously dirty notebooks are unchanged.

## Box scaling and original initialization

The active defaults are `N=10000`, `density=0.5` and seed 1. The original expression `L = int(2*N**0.5/density)` gives a box side of 400; at the original `N=1000` it gives 126. Here `density` is the simulator's scaling parameter, while the actual mean number density is `N/L**2`, equal to 0.0625 in the default run.

[config.json](config.json) contains inputs without a pinned `L`. `resolve_config` returns a copy with the derived integer side; it also accepts a saved resolved configuration, rejecting an explicit `L` that disagrees with the formula. Simulation, trajectory loading, domain placement, world size, moment normalization and saved metadata all use this resolved side. Changing the box size does not change the coarse-field spacing: `cg_res=1` still samples every spatial unit.

[workflow.py](workflow.py) loads the original module through its AST, skipping only the eager `pos`, `phi0` and `vel` assignments. It then sets the configured particle count, density, resolved box and freshly seeded generator, calls the unchanged `init_positions(N)`, and draws uniform headings from the same generator. This retains the original rejection rule, 1.3 minimum separation, 0.5 wall clearance and 200,000-attempt limit. No lattice construction is used. An independent execution of the complete original module with only its `N` assignment changed to 10,000 produced bitwise-identical initial positions and velocities to the saved run.

Alignment strength 5, repulsion strength 1, mass 1, speed 1, characteristic length 1, interaction range 3, regularization 0.02, timestep 0.01, projected constant-speed update, close-encounter cap and reflecting walls are unchanged.

Earlier fixed-box experiments used shuffled hexagonal sites after random placement failed at `L=126`. Those experiments and their artifacts are kept locally; this notebook uses the original initializer at the scaled box size.

## Run

From the repository root, use a Python environment with PySPIDER and the notebook test dependencies (`pytest`, `nbformat`, `nbclient`, `jupyter_client`, and `ipykernel`):

```sh
PYTHONPATH=src NUMBA_NUM_THREADS=2 MPLBACKEND=Agg \
  python -m pytest \
  flocking/updatedparameters/test_workflow.py \
  flocking/updatedparameters/workflow.py \
  flocking/dynamics/test_flocking_data.py \
  flocking/dynamics/test_flocking_quadrature.py \
  --doctest-modules -q
```

This includes a real full-size notebook execution and saves its outputs. Alternatively, select that interpreter in the notebook and run all cells. Each execution creates a unique `runs/<UTC timestamp>_<suffix>/` directory. The notebook requires 10,000 particles, resolved L=400, 61 frames, nonempty fitted equations and exact repeatability as acceptance checks. Function docstrings include NumPy-style runnable examples.

### Reuse an existing trajectory

Set `TRAJECTORY_PATH` in the notebook, or `FLOCKING_TRAJECTORY` in the environment, to a `.npy` path relative to this directory or absolute. Supply `SIMULATION_METADATA_PATH` / `FLOCKING_SIMULATION_METADATA`, or keep `simulation.json` beside the trajectory. Metadata `N`, `L`, `dt` and `PERIODIC_X` must match the resolved configuration; a historical L=126 input is rejected for the current N=10000 defaults.

Rows must be `[step, index, x, y, vx, vy]`, ordered by saved frame and particle index. The existing [trajectory adapter](../dynamics/flocking_data.py) validates and reshapes them. A shortened final interval can be dropped and is recorded; other nonuniform sampling is rejected. Full notebook acceptance requires all 61 uniform frames from 0 to 12. Reuse leaves the source trajectory unchanged, records its relative path and checksum, and runs both analyses without simulating again.

## Coarse-graining and regression

| Setting | Value |
|---|---|
| Particles; resolved box side | 10,000; 400 |
| Scaling density; mean number density | 0.5; 0.0625 |
| Simulation, domain and regression seeds | 1; 1; 1 |
| Duration; timestep; save interval | 12; 0.01; every 20 steps |
| Retained frames; saved-frame spacing | 61; 0.2 |
| Spatial field samples | Uniform, spacing 1 in x and y |
| Kernel | Normalized compact quartic, sigma 4, support radius `sqrt(11)*4` |
| Moment normalization | Divide by `N/L**2` |
| Temporal smoothing; wrapping | None; disabled |
| Domains | 12 seeded unique interior domains |
| Samples per domain | 24 × 24 spatial nodes, 25 saved frames |
| Physical endpoint spans | 23 × 23 × 4.8 |
| Spatial padding | 16, beyond kernel support |
| Weights | `m=6`, `qmax=1`: 8 scalar tensor-product weights |
| Libraries | Scalar and vector; complexity ≤4, at most 2 factors |
| Per-factor derivative limits | Time order ≤1, spatial order ≤2 |
| Column ordering | Sort printed term names before evaluation |
| Integration | Composite trapezoidal on x, y and saved time |
| Column normalization length/time scales | 8; 2 |
| Regression | Power initializer; backward/forward; at most 10 terms, 3 passes |
| Row normalization; training fraction | Unit rows; all rows |
| Residual; jump threshold | Matrix-relative; gamma 1.5, delta 1e-8 |
| Accepted training-residual cutoff | 0.002 |

The unchanged [coordinate-aware adapter](../dynamics/flocking_quadrature.py) evaluates the kernel at the actual uniform nodes and includes physical spacing factors in `dx dy dt`. No quadrature comparison or nonuniform relabeling is performed.

## Executed results: 13 November 2026

The reference run (`20261113T201604_379131Z_6903ec57`, local only) simulated 1,200 steps, retained 61 frames and saved 610,000 rows. The measured initial minimum distance was 1.3001424758983986; the saved speed range was 0.9999999999999997–1.0000000000000002.

Simulation and original initialization took 9.5897 seconds, analysis took 9.9939 seconds, and the independent repeat took 8.9105 seconds. Total measured notebook time was 39.7897 seconds, excluding setup imports before the timer. The scalar matrix is 96 × 17 and the vector matrix is 192 × 15. Both matrices, both column-scale arrays and both dense coefficient arrays matched bitwise, with maximum absolute difference zero; ordered libraries, grids and equation records matched as well.

The accepted relations, in the saved coefficient normalization, were:

- `ρ − 0.9999999999999999 ρ[v_α · v_α] = 0`, training residual 3.364753595965699e-17.
- `0.999974715504884 ∂t ρ + ∂α ρ[v_α] = 0`, training residual 0.001037779569812049.
- `−0.9999999999999998 ρ[v_α] + ρ[v_α · v_β · v_β] = 0`, training residual 1.0665077275133354e-17.

A fresh-kernel reuse run (`20261113T201740_760511Z_a3885e14`, local only) repeated both analyses at `L=400`. All saved matrices, scales and coefficients matched the generated run bitwise. The reference trajectory SHA-256 is `97f1c3e3834ff4bf721052aecb44832db689716d30816eaf9c85e3b16b9a8584`.

The reference test run returned **44 passed**, with existing `FinDiff` deprecation warnings. Tests cover particle-dependent box scaling, seeded initialization, minimum separation, constant speed, direct kernel sums, physical trapezoidal volume, uniform nodes, repeated analysis and the full notebook.

Subsequent width and grid checks support sigma 8, spatial spacing 1, and 95-unit integration spans as working settings for the tested trajectory and library. Those checks are separate from the sigma-4 reproducibility example above; the shipped configuration is unchanged.

## Saved artifacts and interpretation

Each run contains a resolved `config.json`, `simulation.json`, generated `trajectories.npy` (or a reference to reused input), `analysis/` and `repeat/` directories, and consolidated `results.json`. Each analysis directory saves `matrices.npz`, `libraries.json`, `grids.json` and `equations.json`. Arrays include `Q_0`, `Q_1`, `col_weights_0`, `col_weights_1`, `coefficients_0` and `coefficients_1`; dense coefficient rows follow the ordered saved library terms. Read arrays with `numpy.load(path, allow_pickle=False)`. `Q` contains physical weak integrals before regression scaling. Results include source SHA-256 hashes, notebook source-cell hash, interpreter/module versions and paths, Git state and timings.

The fixed-speed moment identities and approximate continuity relation are kinematic diagnostics. They do not identify the force-dependent flocking dynamics. Independent trajectories, richer excitation and sensitivity checks remain necessary before interpreting a fitted relation as a governing dynamical equation.
