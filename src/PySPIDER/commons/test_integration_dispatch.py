from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from PySPIDER.commons.library import ConstantTerm
from PySPIDER.commons.process_library_terms import (
    AbstractDataset,
    IntegrationDomain,
    Weight,
    int_arr,
)
from PySPIDER.commons.truncated_quadrature import truncated_chebyshev_nodes


@dataclass(kw_only=True)
class DummyDataset(AbstractDataset):
    signal: np.ndarray

    def make_libraries(self, **kwargs):
        return None

    def make_domains(self, ndomains, domain_size, pad=0):
        return None

    def eval_prime(self, prime, domain, *args):
        raise NotImplementedError

    def find_scales(self, names=None):
        return None

    def get_char_size(self, term):
        return 1.0

    def eval_term(self, term, domain, debug=False):
        return self.signal


def test_int_arr_default_trapz_matches_numpy():
    vals = np.sin(np.linspace(-1.0, 1.0, 33))
    expected = np.trapezoid(vals)
    actual = int_arr(vals)
    assert np.isclose(actual, expected, atol=1e-14)


def test_int_arr_truncated_grid_unweighted():
    a, b = -0.4, 0.7
    nodes = truncated_chebyshev_nodes(96, a, b)
    vals = np.exp(nodes)
    ref = np.exp(b) - np.exp(a)
    actual = int_arr(
        vals,
        scheme="truncated-grid",
        scheme_options={"nodes": nodes, "interval": (a, b)},
    )
    assert abs(actual - ref) < 1e-10 * (1 + abs(ref))


def test_int_arr_truncated_grid_weight_modes():
    nodes = truncated_chebyshev_nodes(96, -1.0, 1.0)

    jacobi_ref, _ = quad(lambda x: (1.0 - x**2) ** 3.0 * np.cos(x), -1.0, 1.0)
    jacobi_val = int_arr(
        np.cos(nodes),
        scheme="truncated_grid",
        scheme_options={"nodes": nodes, "interval": (-1.0, 1.0), "m": 3.0},
    )
    assert abs(jacobi_val - jacobi_ref) < 1e-9 * (1 + abs(jacobi_ref))

    a, b = -0.3, 0.8
    env_nodes = truncated_chebyshev_nodes(128, a, b)
    env_ref, _ = quad(lambda x: ((x - a) * (b - x)) ** 2.0 * np.exp(x), a, b)
    env_val = int_arr(
        np.exp(env_nodes),
        scheme="truncated-grid",
        scheme_options={"nodes": env_nodes, "interval": (a, b), "envelope_m": 2.0},
    )
    assert abs(env_val - env_ref) < 1e-8 * (1 + abs(env_ref))

    custom_ref, _ = quad(lambda x: np.exp(-x**2) * np.sin(2.0 * x), a, b)
    custom_val = int_arr(
        np.sin(2.0 * env_nodes),
        scheme="truncated-grid",
        scheme_options={
            "nodes": env_nodes,
            "interval": (a, b),
            "weight_func": lambda x: np.exp(-x**2),
        },
    )
    assert abs(custom_val - custom_ref) < 1e-8 * (1 + abs(custom_ref))


def test_dataset_eval_on_domain_uses_selected_scheme():
    a, b = 0.0, 1.0
    nodes = truncated_chebyshev_nodes(80, a, b)
    vals = np.exp(nodes)

    dataset = DummyDataset(
        world_size=[len(nodes)],
        data_dict={},
        observables=[],
        signal=vals,
        integration_scheme="truncated-grid",
        integration_options={"nodes": nodes, "interval": (a, b)},
    )
    dataset.dxs = [1.0]
    domain = IntegrationDomain([0], [len(nodes) - 1])
    weight = Weight(m=[0], q=[0], k=[0], dxs=[1.0])

    result = dataset.eval_on_domain(ConstantTerm(), weight, domain)
    ref = np.exp(b) - np.exp(a)
    assert abs(result - ref) < 1e-10 * (1 + abs(ref))
