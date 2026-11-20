"""Tests for the particle-scaled, uniform-field workflow."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'src'))
import PySPIDER  # Load source package before pytest imports repository __init__.py.


def test_particle_count_resolves_original_box_scaling():
    from workflow import default_config, resolve_config
    config = default_config()
    assert resolve_config(config)['L'] == 400
    config['N'] = 1000
    assert resolve_config(config)['L'] == 126
    config['N'] = 2500
    assert resolve_config(config)['L'] == 200
    config['density'] = 1.0
    assert resolve_config(config)['L'] == 100
    import pytest
    with pytest.raises(ValueError, match='resolved'):
        resolve_config(dict(config, L=126))


def test_full_initialization_preserves_box_spacing_seed():
    assert (HERE / 'workflow.py').is_file(), 'Particle-scaled workflow missing'
    from workflow import initialize_simulator, default_config
    config = default_config()
    simulation = initialize_simulator(config)
    assert simulation['N'] == 10000
    assert simulation['L'] == 400
    assert simulation['seed'] == 1
    p = simulation['pos']
    assert p.shape == (10000, 2)
    assert p.min() >= 0.5 and p.max() <= 399.5
    assert cKDTree(p).query(p, k=2)[0][:, 1].min() >= 1.3
    repeated = initialize_simulator(config)
    np.testing.assert_array_equal(p, repeated['pos'])
    np.testing.assert_array_equal(simulation['vel'], repeated['vel'])
    q, v = simulation['step'](p.copy(), simulation['vel'].copy())
    assert np.isfinite(q).all()
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1, atol=3e-16)


def test_uniform_field_physical_trapezoidal_and_repeatable_analysis(tmp_path):
    import workflow
    assert hasattr(workflow, 'analyze'), 'Independent analysis entry point missing'
    from flocking_data import load_trajectories
    from PySPIDER.commons.library import ConstantTerm, LibraryPrime, DerivativeOrder
    from PySPIDER.discrete.library import CoarseGrainedProduct
    from PySPIDER.commons.process_library_terms import Weight
    config = workflow.default_config()
    config.update(N=1000, T=1.6, kernel_sigma=4.0, cg_res=0.75,
                  domain_width=24.0, domain_frames=9)
    config = workflow.resolve_config(config)
    sim = workflow.initialize_simulator(config)
    path = tmp_path / 'trajectories.npy'
    sim['run_to_T'](config['T'], config['save_every'], str(path))
    data = load_trajectories(path, simulation_dt=config['dt'], box_side=config['L'])
    first = workflow.analyze(data, config, tmp_path / 'first')
    second = workflow.analyze(data, config, tmp_path / 'second')
    for key in first['arrays']:
        np.testing.assert_array_equal(first['arrays'][key], second['arrays'][key])
    assert first['equations'] == second['equations']
    assert first['grids'] == second['grids']
    assert first['libraries'] == second['libraries']
    srd = first['dataset']
    domain = srd.domains[0]
    for axis in range(3):
        nodes = srd._axis_coords(axis, domain)
        np.testing.assert_allclose(np.diff(nodes), srd.dxs[axis])
    weight = Weight(m=[0]*3, q=[0]*3, k=[0]*3, dxs=[1]*3)
    expected = np.prod([(b-a)*dx for a,b,dx in zip(domain.min_corner, domain.max_corner,srd.dxs)])
    assert np.isclose(srd.eval_on_domain(ConstantTerm(), weight, domain), expected)
    prime = LibraryPrime(derivand=CoarseGrainedProduct(observables=()),
                         derivative=DerivativeOrder(x_derivatives=()))
    actual = srd.eval_prime(prime, domain)
    x,y = np.meshgrid(srd._axis_coords(0, domain),srd._axis_coords(1, domain), indexing='ij')
    oracle = np.zeros_like(actual)
    radius2 = 11*config['kernel_sigma']**2
    for t,frame in enumerate(domain.times):
        for point in data['positions'][:,:,frame]:
            oracle[:,:,t] += np.maximum(1-((x-point[0])**2+(y-point[1])**2)/radius2,0)**4
    oracle /= np.pi*radius2/5 * (config['N']/config['L']**2)
    np.testing.assert_allclose(actual, oracle, atol=3e-14)
    assert (tmp_path / 'first' / 'matrices.npz').is_file()


def test_full_notebook_executes_and_compares_actual_10k_analysis():
    path = HERE / '01_updated_flocking.ipynb'
    assert path.is_file(), 'Full-size notebook missing'
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try:
        NotebookClient(notebook, km=km, timeout=1200,
            resources={'metadata':{'path':str(HERE)}}).execute()
    finally:
        nbformat.write(notebook,path)
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    import json
    outputs = '\n'.join(out.get('text','') for cell in notebook.cells
                        for out in cell.get('outputs',[]) if out.output_type == 'stream')
    assert 'FULL_10000_PARTICLE_ACCEPTANCE_PASSED' in outputs
    runs = sorted((HERE/'runs').glob('*/results.json'))
    result = json.loads(runs[-1].read_text())
    assert result['particle_count'] == 10000 and result['box_side'] == 400
    assert result['snapshots_used'] == 61
    assert result['repeatability']['all_equal']
    assert result['equations'], 'Repeatability must include actual fitted coefficients'
    assert result['matrix_shapes'] == [[96,17],[192,15]]
