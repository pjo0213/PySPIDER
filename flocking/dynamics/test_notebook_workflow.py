"""Execute the actual flocking notebook with the current Python kernel."""
from pathlib import Path
import sys


def test_notebook_runs_particle_simulation_and_discrete_discovery():
    path = Path(__file__).parent / "01_flocking_discovery.ipynb"
    assert path.is_file(), "Runnable flocking discovery notebook is missing"
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook = nbformat.read(path, as_version=4)
    # Run the simulator with the notebook's default settings.
    notebook.cells.append(nbformat.v4.new_code_cell(
        "assert run_summary['matrix_shapes']\n"
        "assert run_summary['snapshots_used'] >= 25\n"
        "assert all(np.isfinite(lib.Q).all() for lib in srd.libs.values())\n"
        "assert (RUN_DIR / 'results.json').is_file()\n"
        "assert (RUN_DIR / 'library_matrices.npz').is_file()\n"
        "assert (RUN_DIR / 'simulation.json').is_file()\n"
        "print('END_TO_END_ACCEPTANCE_PASSED', RUN_DIR)\n"
    ))
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=km, timeout=600,
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        # Exclude the acceptance-only extra cell from the user's notebook.
        notebook.cells.pop()
        nbformat.write(notebook, path)
        if km.has_kernel:
            km.shutdown_kernel(now=True)
