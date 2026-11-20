"""Regression checks for the trajectory-to-discrete-SPIDER adapter."""
from pathlib import Path

import numpy as np
import pytest

def test_saved_particle_rows_become_particle_component_time_arrays(tmp_path):
    assert (Path(__file__).parent / "flocking_data.py").is_file(), "Trajectory adapter is missing"
    from flocking_data import load_trajectories

    # Two particles sampled at three saved times.
    rows = np.array([
        [0, 0, 1, 2, 1, 0], [0, 1, 3, 4, 0, 1],
        [20, 0, 1.2, 2, 1, 0], [20, 1, 3, 4.2, 0, 1],
        [40, 0, 1.4, 2, 1, 0], [40, 1, 3, 4.4, 0, 1],
    ], dtype=float)
    path = tmp_path / "trajectories.npy"
    np.save(path, rows)
    data = load_trajectories(path, simulation_dt=0.01, box_side=10)
    np.testing.assert_allclose(data["positions"][0], [[1, 1.2, 1.4], [2, 2, 2]])
    np.testing.assert_allclose(data["velocities"][1], [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_allclose(data["times"], [0, 0.2, 0.4])
    assert data["positions"].shape == (2, 2, 3)
    assert data["deltat"] == pytest.approx(0.2)
    assert data["dropped_final_step"] is None


def _rows(steps=(0, 20, 40, 60)):
    return np.array([[step, i, 1 + i, 2, 1, 0]
                     for step in steps for i in range(2)], dtype=float)


def test_drops_only_the_simulators_short_final_interval(tmp_path):
    from flocking_data import load_trajectories

    path = tmp_path / "trajectory.npy"
    np.save(path, _rows((0, 20, 40, 45)))
    original = path.read_bytes()
    data = load_trajectories(path, simulation_dt=0.01, box_side=10)
    assert data["dropped_final_step"] == 45
    np.testing.assert_array_equal(data["steps"], [0, 20, 40])
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="uniform time spacing"):
        load_trajectories(path, simulation_dt=0.01, box_side=10, drop_irregular_final=False)


@pytest.mark.parametrize("steps", [(0, 20, 45, 60), (0, 20, 40, 80), (0, 20, 20, 40), (0, 20)])
def test_rejects_irregular_or_insufficient_saved_frames(tmp_path, steps):
    from flocking_data import load_trajectories

    path = tmp_path / "trajectory.npy"
    np.save(path, _rows(steps))
    with pytest.raises(ValueError):
        load_trajectories(path, simulation_dt=0.01, box_side=10)


@pytest.mark.parametrize("case", ["nan", "negative_step", "fractional_index", "unordered", "mixed_steps", "incomplete", "out_of_box", "wrong_columns", "empty"])
def test_rejects_malformed_trajectories(tmp_path, case):
    from flocking_data import load_trajectories

    rows = _rows()
    if case == "nan":
        rows[0, 4] = np.nan
    elif case == "negative_step":
        rows[0, 0] = -1
    elif case == "fractional_index":
        rows[0, 1] = 0.5
    elif case == "unordered":
        rows[[0, 1]] = rows[[1, 0]]
    elif case == "mixed_steps":
        rows[1, 0] = 1
    elif case == "incomplete":
        rows = rows[:-1]
    elif case == "out_of_box":
        rows[0, 2] = 11
    elif case == "wrong_columns":
        rows = rows[:, :5]
    elif case == "empty":
        rows = rows[:0]
    path = tmp_path / "trajectory.npy"
    np.save(path, rows)
    with pytest.raises(ValueError):
        load_trajectories(path, simulation_dt=0.01, box_side=10)


@pytest.mark.parametrize("dt, side", [(0, 10), (-1, 10), (float("inf"), 10), (0.01, 0), (0.01, float("nan"))])
def test_requires_valid_physical_metadata(tmp_path, dt, side):
    from flocking_data import load_trajectories

    path = tmp_path / "trajectory.npy"
    np.save(path, _rows())
    with pytest.raises(ValueError):
        load_trajectories(path, simulation_dt=dt, box_side=side)
