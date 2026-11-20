"""Read particles.py trajectory rows as discrete PySPIDER input arrays."""
from pathlib import Path

import numpy as np


def load_trajectories(path, *, simulation_dt, box_side, drop_irregular_final=True):
    """Return positions/velocities shaped (particle, component, saved time).

    ``simulation_dt`` gives the integrator step; saved step indices determine
    the interval between frames. ``box_side`` gives the physical square-box
    side. Supply both from the simulation settings. With the default
    ``drop_irregular_final=True``, discard and report a single shortened final
    interval emitted by particles.py. Other irregular sampling raises
    ValueError. Returned arrays contain retained samples in float64.
    """
    path = Path(path).resolve()
    if not np.isfinite(simulation_dt) or simulation_dt <= 0:
        raise ValueError("simulation_dt must be finite and positive")
    if not np.isfinite(box_side) or box_side <= 0:
        raise ValueError("box_side must be finite and positive")
    rows = np.load(path, allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 6 or len(rows) == 0:
        raise ValueError("Expected nonempty rows [step, index, x, y, vx, vy]")
    if not np.issubdtype(rows.dtype, np.number) or not np.isfinite(rows).all():
        raise ValueError("Trajectory entries must be finite numbers")
    if np.any(rows[:, :2] < 0) or not np.equal(rows[:, :2], np.floor(rows[:, :2])).all():
        raise ValueError("Step and particle indices must be nonnegative integers")
    particle_count = int(rows[:, 1].max()) + 1
    if len(rows) % particle_count:
        raise ValueError("Incomplete particle snapshot")
    snapshots = rows.reshape(-1, particle_count, 6)
    steps = snapshots[:, 0, 0].astype(np.int64)
    if not np.equal(snapshots[:, :, 0], steps[:, None]).all():
        raise ValueError("Each snapshot must have one common step number")
    if not np.equal(snapshots[:, :, 1], np.arange(particle_count)).all():
        raise ValueError("Rows must be ordered by snapshot, then particle index")
    if len(steps) < 3 or np.any(np.diff(steps) <= 0):
        raise ValueError("At least three strictly increasing snapshots are required")
    if np.any(snapshots[:, :, 2:4] < 0) or np.any(snapshots[:, :, 2:4] > box_side):
        raise ValueError("Positions must lie within [0, box_side]")
    gaps = np.diff(steps)
    dropped_final_step = None
    if not np.equal(gaps, gaps[0]).all():
        if (drop_irregular_final and len(steps) >= 4
                and np.equal(gaps[:-1], gaps[0]).all()
                and 0 < gaps[-1] < gaps[0]):
            dropped_final_step = int(steps[-1])
            snapshots, steps = snapshots[:-1], steps[:-1]
        else:
            raise ValueError("Saved snapshots must have uniform time spacing")
    return {
        "positions": np.ascontiguousarray(snapshots[:, :, 2:4].transpose(1, 2, 0), dtype=np.float64),
        "velocities": np.ascontiguousarray(snapshots[:, :, 4:6].transpose(1, 2, 0), dtype=np.float64),
        "steps": steps,
        "times": steps * float(simulation_dt),
        "deltat": float((steps[1] - steps[0]) * simulation_dt),
        "particle_count": particle_count,
        "box_side": float(box_side),
        "dropped_final_step": dropped_final_step,
    }
