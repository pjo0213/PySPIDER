"""Reproduce the 5,000-particle reflecting-box run with doubled repulsion."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy
from scipy.spatial import cKDTree

_PARTICLES = Path(__file__).resolve().parents[2] / 'particles.py'
_N, _L, _END, _STRIDE = 5000, 200, 40000, 50
_RC_ALIGN = 6.0
_PHYSICS = dict(a=1.0, v=1.0, m=1, beta=5.0, eps=2.0, rc=3.0, r_min=0.02,
                dt=0.01, PERIODIC_X=False)


def _split_force(min_image):
    # Keep the archived pair ordering and reduction, including the addition
    # before bincount. Summing the two force components separately changes rounding.
    rc_rep, a, eps, beta, floor = (_PHYSICS[k] for k in ('rc', 'a', 'eps', 'beta', 'r_min'))

    def forces(pos, vel):
        pairs = cKDTree(pos).query_pairs(_RC_ALIGN, output_type='ndarray')
        F = np.zeros_like(pos)
        if not len(pairs):
            return F
        i, j = pairs[:, 0], pairs[:, 1]
        d = min_image(pos[i] - pos[j])
        r = np.linalg.norm(d, axis=1)
        K = beta * (1.0 - r / _RC_ALIGN) ** 2
        gap = np.maximum(r - a, floor)
        rep = 4 * eps * a * np.maximum(rc_rep - r, 0.0) / gap**3
        f = K[:, None] * (vel[j] - vel[i]) + (rep / r)[:, None] * d
        for k in (0, 1):
            F[:, k] = np.bincount(i, f[:, k], len(pos)) - np.bincount(j, f[:, k], len(pos))
        return F

    return forces


def _initialize(seed):
    # Load the original numerical functions without eager initialization or
    # plotting, with globals local to this run (no change to simulator defaults).
    source = ast.parse(_PARTICLES.read_bytes(), filename=str(_PARTICLES))
    names = {'min_image', 'init_positions', 'step'}
    source.body = [node for node in source.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    rng = np.random.default_rng(seed)
    ns: dict[str, Any] = dict(_PHYSICS, np=np, rng=rng, N=_N, L=_L)
    exec(compile(source, str(_PARTICLES), 'exec'), ns)
    ns['forces'] = _split_force(ns['min_image'])
    p = ns['init_positions'](_N)
    phi = rng.uniform(0, 2 * np.pi, _N)
    w = ns['v'] * np.c_[np.cos(phi), np.sin(phi)]
    return p, w, ns['step']


def _run(seed, end, out):
    out.mkdir(parents=True, exist_ok=False)
    p, w, step = _initialize(seed)
    schedule = np.arange(0, end + 1, _STRIDE, dtype=np.int64)
    if schedule[-1] != end:
        schedule = np.append(schedule, np.int64(end))
    partial = out / 'states.partial.npy'
    states = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float64,
                                       shape=(len(schedule), _N, 4))
    n = 0
    for frame, target in enumerate(schedule):
        while n < target:
            p, w = step(p, w)
            n += 1
        if not (np.isfinite(w).all() and p.min() >= 0 and p.max() <= _L):
            raise RuntimeError(f'Nonfinite or out-of-box state at step {n}')
        states[frame, :, :2], states[frame, :, 2:] = p, w
    states.flush()
    del states
    partial.rename(out / 'states.npy')
    np.save(out / 'steps.npy', schedule, allow_pickle=False)
    metadata = dict(_PHYSICS, seed=seed, N=_N, L=_L, alignment_cutoff=_RC_ALIGN,
                    repulsion_cutoff=_PHYSICS['rc'], last_step=end, frames=len(schedule),
                    saved_stride=_STRIDE, initializer='original_random_sequential',
                    python=platform.python_version(), architecture=platform.machine(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    particles_sha256=hashlib.sha256(_PARTICLES.read_bytes()).hexdigest())
    (out / 'simulation.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Seed {seed}: saved {len(schedule)} frames through step {end} '
          f'(t={end * _PHYSICS["dt"]:g}) to {out}')


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, choices=(1, 2))
    parser.add_argument('--steps', type=int, default=_END,
                        help='number of updates; default: %(default)s (t=400)')
    parser.add_argument('--out', type=Path, required=True, help='a new output directory')
    args = parser.parse_args()
    if not 0 <= args.steps <= _END:
        parser.error(f'--steps must be between 0 and {_END}')
    try:
        _run(args.seed, args.steps, args.out)
    except FileExistsError:
        parser.error(f'Output directory already exists or has a file in its path: {args.out}')


if __name__ == '__main__':
    _main()
