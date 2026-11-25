# Selected flocking regime: doubled repulsion

This run uses 5,000 particles in a 200 × 200 box with four reflecting walls. In a matched two-seed comparison, doubling repulsion reduced density depletion while retaining stronger time variation than shortening the alignment range. This directory adds a simulation entry point for that regime. The [10,000-particle notebook workflow](../README.md) and its defaults remain unchanged.

## Run

From the repository root, use Python with NumPy and SciPy:

```sh
D=flocking/updatedparameters/selected_regime
python "$D/simulate.py" 1 --out "$D/runs/seed1"
python "$D/simulate.py" 2 --out "$D/runs/seed2"
# Short run and regression test:
python "$D/simulate.py" 1 --steps 200 --out "$D/runs/check-seed1"
python -m unittest discover -s "$D" -v
```

Each output directory must be new. The default is 40,000 updates at `dt=0.01`, through `t=400`. `steps.npy` contains integer update numbers: 0, every 50th step (spacing 0.5 in time), and the requested final step if it misses the stride. `states.npy` stores float64 `(frame, particle, 4)` in `(x, y, vx, vy)` order, with the same particle index in every frame. A full run has 801 frames and uses approximately 128 MB per seed. Use `np.load(path, mmap_mode='r', allow_pickle=False)` to read states without loading the entire file. `simulation.json` records the settings and numerical dependency versions. If a run stops during integration, it leaves `states.partial.npy`; use a new directory to rerun it.

[`simulate.py`](simulate.py) uses the original random sequential placement and `step` from [`particles.py`](../../particles.py) in a separate namespace. The seed drives both placement and the subsequent uniform headings. The placement rule keeps centers at least `1.3a` apart with `0.5a` wall clearance. The force uses separate cutoffs: alignment weight `5(1 − r/6a)²` within `6a`, and the original repulsion profile within `3a`, with amplitude `eps=2` and gap floored at `0.02a`. The study's split-force implementation uses a single `cKDTree` pair query at the larger cutoff and adds the two pair contributions before accumulating them. The update retains `a=m=v=1`, the perpendicular force projection, the `0.25v` cap on the velocity increment, speed renormalization, and specular reflection. The cap was active for about 36% of particles on the first update in these two runs; the archived comparison uses this capped dynamics at the original timestep.

## Matched comparison through t=400

All three cases share the initial state within each seed. The reference uses `eps=1` and alignment cutoff `6a`; the shorter-alignment control uses `eps=1` and cutoff `4.5a` at fixed `beta=5`.

| Case | Seed | Density CV | Coverage | Depleted area | Five-unit flow change | 20-unit net distance, median / p10 |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 1 | 0.308 | 93.8% | 2.23% | 0.227 | 19.68 / 15.87 |
| Reference | 2 | 0.292 | 96.0% | 1.23% | 0.342 | 19.64 / 13.37 |
| Doubled repulsion | 1 | 0.242 | 97.7% | 0.13% | 0.440 | 19.11 / 10.53 |
| Doubled repulsion | 2 | 0.202 | 99.4% | 0.00% | 0.419 | 18.81 / 9.53 |
| Shorter alignment | 1 | 0.200 | 99.7% | 0.00% | 0.176 | 19.85 / 17.96 |
| Shorter alignment | 2 | 0.195 | 100.0% | 0.00% | 0.166 | 19.84 / 18.09 |

Field summaries are time medians on a 68 × 68 interior grid of spacing 2 with quartic smoothing at `sigma=8`, using 121 matched times from 100 through 400 at spacing 2.5. Density CV is the interior spatial standard deviation divided by the interior mean. Coverage is the fraction of cells at or above half the nominal density `N/L²`; depletion is the fraction below one quarter. Five-unit flow change is the RMS Eulerian velocity-vector difference on cells at or above quarter density at both endpoints. Net distance is the lab-frame displacement of the same particles over 20 units. Its median and 10th percentile pool all particles across the 561 overlapping windows wholly within `t=100–400`.

Doubling repulsion increased five-unit flow change in both seeds and retained more than 95% of the reference median net displacement, although the low-displacement tail shifted substantially. Shortening alignment gave more even coverage but smaller five-unit changes than the reference. These are screening results for two matched seeds. Source: `comparison.csv` from the local study `box200-anti-clumping-20261120T235714Z` (SHA-256 `c0ff729e8f8e781894ec6c7defeb68f8d94f793ee722e0d3d076d0373217ffe6`). The entry point generates the selected trajectories; the historical comparison data and analysis scripts are not required to run it.

## Reproduction evidence

[`test_simulate.py`](test_simulate.py) exercises the CLI for both seeds and checks archived sample rows at steps 0, 50 and 100 without requiring the original arrays. A separate run from a clean repository snapshot matched all 5,000 particles bitwise at steps 0, 50, 100, 150 and 200 for both seeds. Verification and the archived reference used Python 3.14.7, NumPy 2.3.5 and SciPy 1.18.0 on macOS arm64. Changes to floating-point operations or neighbor ordering can affect bitwise replay across environments.

Both full runs also reproduced the archived `states.npy` files byte for byte:

| Seed | SHA-256 |
|---|---|
| 1 | `15df73c7a6df9bd5a2ac7507bab67fcaf185d8c3da3fded7124392a8efad637a` |
| 2 | `8bdc2ca20cc7f4617c6549d3bf8ce77742c227412db5305ca7130df6ba83848e` |

## What the force-law tests established

The table above describes simulated behavior. Weak-window geometry and saved-time spacing still need checking on this regime before inference. In a separate 10,000-particle, `L=400`, `eps=1` study, native rank-one PySPIDER used projected force observables and a restricted library. At saved spacing 0.20, two matched windows gave a 32 × 11 matrix per seed. The strict competitive search returned four proxy-only near-dependencies (lowest training residual `1.93e-5`); none contained the momentum time derivative. No model passed the balance and transfer screen. Native assembly at spacing 0.05 was interrupted before producing a discovery result. This evidence comes from the local `native-20261120T201707Z` study (`discovery-20.json`, SHA-256 `8572f05cf871a524e8a12f3706659647939826ae259968a75edc46a641b391ef`).

The targeted consecutive-step heading fit in that earlier regime recovered the known inverse-cube gap template when supplied among the candidates. Separate weak fits at finer cadence could not clearly distinguish the cubic from a mixture omitting it. Neither result establishes unique discovery of the radial law, and neither was performed on the selected trajectory here.
