"""Execute the quadrature notebook on saved flocking trajectories."""
from pathlib import Path
import sys


def test_quadrature_notebook_runs_all_rules():
    path = Path(__file__).parent / '02_quadrature_comparison.ipynb'
    assert path.is_file(), 'Quadrature comparison notebook is missing'
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook = nbformat.read(path, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(
        "assert len(comparison_records) == 4\n"
        "assert {r['spatial_rule'] for r in comparison_records} == {'trapezoidal', 'moment-matching', 'clenshaw-curtis', 'chebyshev-gauss'}\n"
        "assert all(np.isfinite(r['continuity_coefficient']) for r in comparison_records)\n"
        "assert all(np.isfinite(r['continuity_balance_residual']) for r in comparison_records)\n"
        "assert all(r['matrix_shapes'] == [[96, 17], [192, 15]] for r in comparison_records)\n"
        "assert (RUN_DIR / 'comparison.json').is_file()\n"
        "assert hashlib.sha256(trajectory_path.read_bytes()).hexdigest() == trajectory_hash\n"
        "print('QUADRATURE_NOTEBOOK_VERIFIED', RUN_DIR)\n"
    ))
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try:
        NotebookClient(notebook, km=km, timeout=600,
            resources={'metadata': {'path': str(path.parent)}}).execute()
    finally:
        notebook.cells.pop()
        nbformat.write(notebook, path)
        if km.has_kernel:
            km.shutdown_kernel(now=True)
