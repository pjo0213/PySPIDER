# Flocking dynamics notebooks

Select the PySPIDER Conda interpreter and run the notebooks in order:

1. [01_flocking_discovery.ipynb](01_flocking_discovery.ipynb) generates or loads trajectories and runs an initial discrete-SPIDER analysis using the default trapezoidal rule.
2. [02_quadrature_comparison.ipynb](02_quadrature_comparison.ipynb) applies moment matching, Clenshaw–Curtis, and Chebyshev–Gauss quadrature to saved trajectories, alongside a trapezoidal baseline.

Use an environment with PySPIDER and the notebook dependencies installed.

## Inputs

- Default `TRAJECTORY_PATH = None`: run the supplied [particles.py](../particles.py), with its existing simulation parameters, for 12 simulation-time units and save every 20 steps. This produces 1,000 particles and 61 snapshots under the current defaults.
- Existing data: set `TRAJECTORY_PATH = '../trajectories.npy'` relative to this directory, and set `EXISTING_DT`, `EXISTING_L`, and `EXISTING_PERIODIC_X` to the original simulation settings. The saved step indices and integrator timestep determine the interval between frames.
- Input rows are `[step, index, x, y, vx, vy]`, ordered by frame and particle index. The adapter checks ordering, finite values, bounds, and sampling. One short final saving interval can be dropped explicitly; other nonuniform sampling is rejected.
- [coarse_grain.py](../coarse_grain.py) and [visualize_fields.py](../visualize_fields.py) produce separate visualization fields. The notebooks pass raw trajectories to discrete PySPIDER for coarse-graining during library evaluation.

## Discovery and outputs

The notebooks contain dataset construction, library evaluation, and regression calls. `flocking_data.py` validates and reshapes input data. `flocking_quadrature.py` supplies spatial evaluation nodes and physical integration intervals for the comparison.

The initial scalar/vector libraries use complexity 4 and at most two coarse-grained factors per term. Each factor can have at most one time derivative and two spatial derivatives; a product can contain derivatives on both factors. The parameter cells control these restrictions, normalization scales, and the residual cutoff.

The first notebook creates a unique `runs/<UTC timestamp>_<suffix>/` directory containing:

- `trajectories.npy` when data are generated;
- `simulation.json` with simulation settings and provenance;
- `library_matrices.npz` with unscaled weak matrices and column scales;
- `libraries.json` with ordered term names and matrix shapes;
- `results.json` with settings, data/source hashes, integration-domain corners, coefficients, and normalized training residuals.

Use `numpy.load(..., allow_pickle=False)` for arrays and `json.load(...)` for records. Generated runs and caches are ignored. The versioned notebooks are distributed without cell outputs.

## Quadrature comparison

The second notebook selects the latest saved trajectory with a sibling `simulation.json`, or a file specified by `SOURCE_TRAJECTORY`. Every case uses the same integration supports and ordered library terms. The comparison keeps the trapezoidal case's column scales for regression.

| Spatial rule | Spatial samples | Time rule |
|---|---|---|
| `trapezoidal` | Uniform | `trapezoidal` |
| `moment-matching` | Uniform | `moment-matching` |
| `clenshaw-curtis` | Full mapped Lobatto grid per domain | `moment-matching` |
| `chebyshev-gauss` | Full mapped Chebyshev–Gauss grid per domain, with Fejér-I integration weights | `moment-matching` |

The particle kernel is evaluated directly at each spatial grid's coordinates. Time uses the original saved snapshots. `MOMENT_DEGREE=10` limits the requested polynomial moments on uniform grids. Each integral includes the physical space/time measure: trapezoidal axes receive their sample-spacing factors, while the other rules include interval Jacobians.

Results go to a new `runs/quadrature_<UTC timestamp>_<suffix>/` directory. `comparison.json` records settings, source/data hashes, continuity diagnostics, and discovered equations. Each rule also saves `*_matrices.npz` and `*_grids.json` with its actual nodes and integration supports.

## Interpretation and limits

All rows enter the regression fit. Fixed particle speed imposes density/velocity-moment identities, and force projection changes the momentum balance. Assess inferred dynamics on independent trajectories and check sensitivity to timestep, saving interval, kernels, spatial resolution, domains, and library choices.

Interior integration domains have padding of four kernel-sigma units. With `wrap=False`, their kernel supports stay inside the box even when x is periodic. Both notebooks use serial library evaluation and retain the original time samples without temporal smoothing.

Reading: [dissertation §2.4 and §5.2](../../DG_Dissertation.pdf), and [discrete tutorial](../../tutorials/02_Discrete.ipynb).

## Verification

From this directory with the PySPIDER interpreter selected:

```sh
python -m pytest test_flocking_data.py test_notebook_workflow.py test_flocking_quadrature.py test_quadrature_notebook.py -q
```

The notebook tests start kernels with the invoking interpreter, execute the analysis, check saved artifacts, and save executed outputs. The quadrature tests check actual node locations, kernel values, physical-volume integrals, and weighted tensor integrals against independent Gauss–Legendre integration. Every notebook execution creates a new run directory.
