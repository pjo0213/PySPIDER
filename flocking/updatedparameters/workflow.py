"""Particle-scaled flocking and reproducible uniform-grid weak regression."""
from pathlib import Path
import ast
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(HERE.parent / 'dynamics'))


def default_config():
    """Return explicit simulation and analysis settings.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        JSON-serializable inputs; ``resolve_config`` derives the box side.

    Examples
    --------
    >>> default_config()['N'], default_config()['density']
    (10000, 0.5)
    """
    return dict(N=10000, density=0.5, seed=1, initialization='original_random_sequential',
                minimum_separation=1.3, T=12.0,
                dt=0.01, save_every=20, periodic_x=False,
                domain_seed=1, regression_seed=1, kernel_sigma=4.0,
                cg_res=1.0, domain_width=24.0, domain_frames=25,
                n_domains=12, weight_m=6, weight_qmax=1,
                max_complexity=4, max_rho=2, max_dt=1, max_dx=2,
                include_rank2=False, L_scale=8.0, T_scale=2.0,
                spatial_rule='trapezoidal', time_rule='trapezoidal',
                time_sigma=0, wrap=False, discovery_threshold=2e-3,
                regression_max_k=10, regression_max_passes=3,
                regression_max_equations=12)


def resolve_config(config):
    """Resolve the original particle-dependent box side without changing input.

    Parameters
    ----------
    config : dict
        Particle count ``N`` and positive dimensionless ``density`` parameter.
        An optional saved ``L`` must agree with the original scaling formula.

    Returns
    -------
    dict
        Copied settings with integer ``L = int(2*N**0.5/density)``. This box
        side is independent of coarse-field spacing ``1/cg_res``.

    Examples
    --------
    >>> resolve_config(default_config())['L']
    400
    >>> resolve_config(dict(default_config(), N=1000))['L']
    126
    """
    resolved = dict(config)
    n, density = config['N'], config['density']
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('N must be a positive integer')
    if not np.isfinite(density) or density <= 0:
        raise ValueError('density must be positive and finite')
    side = int(2*n**0.5/density)
    if side < 1 or ('L' in config and config['L'] != side):
        raise ValueError('L must agree with resolved original particle-dependent box side')
    resolved['L'] = side
    return resolved


def initialize_simulator(config):
    """Initialize the original simulator at its particle-dependent box size.

    Parameters
    ----------
    config : dict
        Settings returned by ``default_config`` or ``resolve_config``. The
        original random sequential ``init_positions`` uses configured N and a
        freshly seeded generator; headings continue from that same generator.

    Returns
    -------
    dict
        Simulator namespace containing original force, step, and save routines.
        Only eager initial arrays and configured globals are replaced; the
        original initializer and its 200,000-attempt limit are unchanged.

    Examples
    --------
    >>> simulation = initialize_simulator(default_config())
    >>> simulation['pos'].shape
    (10000, 2)
    """
    config = resolve_config(config)
    path = HERE.parent / 'particles.py'
    tree = ast.parse(path.read_text())
    # Skip original eager random placement and headings, retaining definitions.
    excluded = {'pos', 'phi0', 'vel'}
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id in excluded for t in node.targets))]
    ns = {'__name__': 'particle_scaled_particles', '__file__': str(path)}
    exec(compile(tree, str(path), 'exec'), ns)

    if config['dt'] != ns['dt'] or config['periodic_x'] != ns['PERIODIC_X']:
        raise ValueError('dt and boundary conditions must match supplied simulator')
    if config['initialization'] != 'original_random_sequential':
        raise ValueError('Use original_random_sequential initialization')
    if config['minimum_separation'] != 1.3 * ns['a']:
        raise ValueError('Preserve the original 1.3a minimum separation')
    rng = np.random.default_rng(config['seed'])
    ns.update(N=config['N'], density=config['density'], L=config['L'],
              seed=config['seed'], rng=rng)
    p = ns['init_positions'](config['N'])
    phi = rng.uniform(0, 2*np.pi, config['N'])
    ns.update(N=config['N'], L=config['L'], seed=config['seed'], rng=rng,
              pos=p, phi0=phi, vel=ns['v']*np.c_[np.cos(phi), np.sin(phi)])
    return ns


def analyze(trajectory, config, output_dir):
    """Sample continuous particle fields and assemble trapezoidal weak matrices.

    Parameters
    ----------
    trajectory : dict
        Validated ``flocking_data.load_trajectories`` output.
    config : dict
        Explicit settings from ``default_config``. Space and time rules must
        both be trapezoidal. Domains are drawn with ``domain_seed`` and terms
        sorted by their printed names before any matrix evaluation.
    output_dir : path-like
        New directory for matrices, coefficients, grids, and ordered libraries.

    Returns
    -------
    dict
        Dataset, arrays, ordered libraries, grids, relations and wall runtime.
        Arrays include physical-measure matrices, column scales and dense
        equation coefficients aligned with each saved library's term order.

    Examples
    --------
    >>> import contextlib, io, tempfile
    >>> from flocking_data import load_trajectories
    >>> cfg = default_config()
    >>> cfg.update(N=1000, T=1.6, domain_frames=9, cg_res=0.75)
    >>> cfg = resolve_config(cfg)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     with contextlib.redirect_stdout(io.StringIO()):
    ...         sim = initialize_simulator(cfg)
    ...         path = Path(folder) / 'trajectories.npy'
    ...         sim['run_to_T'](cfg['T'], cfg['save_every'], str(path))
    ...         data = load_trajectories(path, simulation_dt=cfg['dt'], box_side=cfg['L'])
    ...         result = analyze(data, cfg, Path(folder) / 'analysis')
    ...     print([library['shape'] for library in result['libraries']])
    [[96, 17], [192, 15]]
    """
    import json
    from time import perf_counter
    from flocking_quadrature import QuadratureSRDataset
    from PySPIDER.commons.library import Observable
    from PySPIDER.commons.identify_models import interleave_identify
    from PySPIDER.commons.sparse_reg_bf import Scaler, Initializer, ModelIterator, Residual, Threshold

    config = resolve_config(config)
    started = perf_counter()
    output_dir = Path(output_dir)
    if config['spatial_rule'] != 'trapezoidal' or config['time_rule'] != 'trapezoidal':
        raise ValueError('This workflow requires trapezoidal integration in space and time')
    if config['wrap'] or config['time_sigma'] != 0:
        raise ValueError('Use interior nonwrapping domains without temporal smoothing')
    n, _, nt = trajectory['positions'].shape
    if n != config['N'] or trajectory['box_side'] != config['L']:
        raise ValueError('Trajectory particle count and box must match config')
    padding = 4*config['kernel_sigma']
    if (config['kernel_sigma'] <= 0 or config['cg_res'] <= 0
            or config['domain_width'] + 2*padding > config['L']
            or config['domain_frames'] > nt or config['domain_frames'] < 8
            or int(config['domain_width']*config['cg_res']) < 8
            or config['n_domains'] < 1 or config['L_scale'] <= 0 or config['T_scale'] <= 0):
        raise ValueError('Positive scales and at least eight samples per axis must fit padded data')
    srd = QuadratureSRDataset(world_size=np.array([config['L'],config['L'],nt]),
        particle_pos=trajectory['positions'], data_dict={'v':trajectory['velocities']},
        observables=[Observable(string='v',rank=1)],
        irreps=QuadratureSRDataset.all_rank2_irreps() if config['include_rank2'] else QuadratureSRDataset.all_rank2_irreps()[:2],
        kernel_sigma=config['kernel_sigma'], cg_res=config['cg_res'], deltat=trajectory['deltat'],
        rho_scale=n/config['L']**2, time_sigma=0, wrap=False,
        spatial_rule='trapezoidal', time_rule='trapezoidal',
        cache_primes=True, cleanup_cache=False)
    srd.make_libraries(**{key:config[key] for key in ['max_complexity','max_rho','max_dt','max_dx']})
    for lib in srd.libs.values():
        lib.terms.sort(key=str)
    np.random.seed(config['domain_seed'])
    srd.make_domains(ndomains=config['n_domains'],
        domain_size=[config['domain_width'],config['domain_width'],config['domain_frames']],pad=padding)
    if len({(tuple(d.min_corner),tuple(d.max_corner)) for d in srd.domains}) != config['n_domains']:
        raise ValueError('Duplicate domains; change domain_seed')
    srd.make_weights(m=config['weight_m'],qmax=config['weight_qmax'])
    srd.set_LT_scale(L=config['L_scale'],T=config['T_scale'])
    srd.make_library_matrices(parallel=False)
    libs = [srd.libs[i] for i in srd.irreps]
    arrays, records, options = {}, [], []
    for i,lib in enumerate(libs):
        scales = np.asarray(lib.col_weights)
        if not np.isfinite(lib.Q).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Nonfinite matrix or nonpositive column scale')
        if lib.Q.shape[0] <= lib.Q.shape[1]:
            raise ValueError('Increase domains/weights for overdetermined libraries')
        arrays[f'Q_{i}'] = lib.Q.copy()
        arrays[f'col_weights_{i}'] = scales.copy()
        records.append(dict(irrep=str(lib.irrep), key=f'Q_{i}',shape=list(lib.Q.shape),terms=[str(t) for t in lib.terms]))
        k = min(config['regression_max_k'],len(lib.terms))
        options.append(dict(scaler=Scaler(sub_inds=None,char_sizes=scales.copy(),row_norms=None,unit_rows=True,train_fraction=1),
            initializer=Initializer(method='power',start_k=k),
            model_iterator=ModelIterator(max_k=k,backward_forward=True,max_passes=config['regression_max_passes']),
            residual=Residual(residual_type='matrix_relative'),
            threshold=Threshold(threshold_type='jump',gamma=1.5,delta=1e-8),verbose=False))
    np.random.seed(config['regression_seed'])
    equations,residuals,_,_,_ = interleave_identify(libs,options,
        threshold=config['discovery_threshold'],max_equations=config['regression_max_equations'],report_accuracy=True)
    equation_records = []
    for eq,res in zip(equations,residuals):
        equation_records.append(dict(equation=eq.pstr(),terms=[str(t) for t in eq.terms],
                                     coefficients=[float(c) for c in eq.coeffs],training_residual=float(res)))
    for i,lib in enumerate(libs):
        dense = []
        for eq in equations:
            if eq.rank == lib.irrep.rank:
                row = np.zeros(len(lib.terms))
                for term,c in zip(eq.terms,eq.coeffs):
                    row[lib.terms.index(term)] = c
                dense.append(row)
        arrays[f'coefficients_{i}'] = np.asarray(dense).reshape(-1,len(lib.terms))
    grids = [dict(min_corner=[int(v) for v in d.min_corner],max_corner=[int(v) for v in d.max_corner],
        nodes=[srd._axis_coords(i,d).tolist() for i in range(3)],
        support=[[float(d.min_corner[i]*srd.dxs[i]),float(d.max_corner[i]*srd.dxs[i])] for i in range(3)]) for d in srd.domains]
    output_dir.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output_dir/'matrices.npz',**arrays)
    for name,value in [('libraries',records),('grids',grids),('equations',equation_records)]:
        (output_dir/(name+'.json')).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return dict(dataset=srd,arrays=arrays,libraries=records,grids=grids,
                equations=equation_records,seconds=perf_counter()-started)
