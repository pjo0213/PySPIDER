"""CLI regression against rows sampled from the doubled-repulsion archive."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).with_name('simulate.py').resolve()
# Seed -> particle 0 at steps 0, 50 and 100, in (x, y, vx, vy) order.
REFERENCE = {
    1: [
        [102.35250331535109, 189.64227556886112, 0.005646859272504224, -0.9999840563630785],
        [102.34393499537138, 189.15141724750458, 0.5052169073131982, -0.8629923965857911],
        [102.74076816605374, 189.26905571990835, 0.5565411730544111, 0.8308200302684209],
    ],
    2: [
        [52.560814715613965, 59.89973753941054, 0.9717754521172253, -0.23590775880068557],
        [52.993799808777126, 59.66481745536431, 0.9858427113483922, -0.1676727422130701],
        [53.48746806709661, 59.72253386010539, 0.9815233395948695, 0.19134245172082054],
    ],
}


class SelectedRegimeTest(unittest.TestCase):
    def test_short_runs_reproduce_archived_rows(self):
        for seed, reference in REFERENCE.items():
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'run'
                cmd = [sys.executable, str(SCRIPT), str(seed), '--steps', '101', '--out', str(out)]
                result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                steps = np.load(out / 'steps.npy', allow_pickle=False)
                states = np.load(out / 'states.npy', allow_pickle=False)
                self.assertEqual(steps.dtype, np.int64)
                self.assertEqual(states.dtype, np.float64)
                np.testing.assert_array_equal(steps, [0, 50, 100, 101])
                self.assertEqual(states.shape, (4, 5000, 4))
                np.testing.assert_allclose(states[:3, 0], reference, rtol=0, atol=1e-9)
                np.testing.assert_allclose(np.linalg.norm(states[:, :, 2:], axis=2), 1,
                                           rtol=0, atol=1e-14)
                self.assertTrue(np.all((0 <= states[:, :, :2]) & (states[:, :, :2] <= 200)))
                meta = json.loads((out / 'simulation.json').read_text())
                self.assertEqual((meta['seed'], meta['dt'], meta['last_step']), (seed, 0.01, 101))
                occupied = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
                self.assertNotEqual(occupied.returncode, 0)
                self.assertIn('Output directory already exists', occupied.stderr)
                np.testing.assert_array_equal(np.load(out / 'states.npy', allow_pickle=False), states)


if __name__ == '__main__':
    unittest.main()
