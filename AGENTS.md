# PySPIDER

PySPIDER is a pure-Python scientific library (the SPIDER framework) for data-driven
discovery of symmetry-equivariant PDE / integro-differential models via sparse regression.
There is **no web server, service, database, or GUI application** — the "application" is the
importable `PySPIDER` package plus the demonstration notebooks in `tutorials/`.

## Cursor Cloud specific instructions

### Environment
- This VM is Ubuntu with a PEP 668 "externally managed" system Python 3.12, so dependencies
  must live in a virtualenv. The startup update script creates/refreshes `.venv/` at the repo
  root and installs the package editable (`pip install -e .`) plus dev/tutorial extras
  (`matplotlib jupyter nbconvert ipykernel h5py pytest`). Always use the `.venv` interpreter,
  e.g. `.venv/bin/python`, `.venv/bin/jupyter`.
- `python3.12-venv` (system apt package) is required for venv creation and is already installed
  in the environment/snapshot. It is intentionally **not** in the update script (no system deps
  there); if venv creation ever fails with an `ensurepip` error, run
  `sudo apt-get install -y python3.12-venv`.

### Tests & lint
- There is **no automated test suite and no lint config** in this repo. Do not expect
  `pytest` to collect anything: the only `*_test.py` file
  (`src/PySPIDER/commons/greedy_rounding_test.py`) is a NotebookLM-generated script that
  imports the commercial `gurobipy` package and is explicitly excluded from packaging
  (`MANIFEST.in`) — ignore it. Correctness is demonstrated via the tutorial notebooks.

### Running / demonstrating the product
- The canonical end-to-end demo is `tutorials/01_Continuous.ipynb` (continuous systems) and
  `tutorials/02_Discrete.ipynb` (discrete/particle systems). Both **generate their own
  synthetic datasets in-notebook** (Taylor-Green vortex / Vicsek-like particles), so they need
  no external data files. Real datasets live under gitignored `src/PySPIDER/*/data/` dirs and
  are not in the repo.
- Run a notebook headless (good for CI-style verification):
  `.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 tutorials/01_Continuous.ipynb --output /tmp/out.ipynb`
  (the continuous tutorial finishes in ~15s). A successful run of `01_Continuous.ipynb`
  discovers incompressibility `∂α u_α = 0` and the Navier-Stokes momentum balance
  `∂α p + u_β·∂β u_α = 0` with residuals ~1e-15.
- Run interactively: `.venv/bin/jupyter notebook --no-browser --ip=127.0.0.1 --port=8888`.
- The notebooks prepend `../src` to `sys.path`, but since the package is installed editable the
  `PySPIDER.*` imports resolve to the local source either way.
- Model-matrix assembly uses `multiprocessing` (`make_library_matrices(parallel=True,
  num_processors=N)`); keep `num_processors` small (2) on constrained VMs.
