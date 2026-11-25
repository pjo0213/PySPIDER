"""Small public quadrature regressions; run with Python's unittest discovery."""

import unittest

import numpy as np

from PySPIDER.commons.integration import int_arr
from PySPIDER.commons.library import DerivativeOrder, LibraryPrime, LibraryTerm, Observable
from PySPIDER.commons.process_library_terms import IntegrationDomain, Weight
from PySPIDER.commons.quadrature_schemes import (
    chebyshev_gauss_weights, clenshaw_curtis_weights, mapped_chebyshev_gauss_nodes,
    mapped_chebyshev_nodes, moment_matched_quad_weights, truncated_chebyshev_nodes,
    truncated_chebyshev_gauss_nodes,
)
from PySPIDER.continuous.process_library_terms import SRDataset


class QuadraturePublicAPI(unittest.TestCase):
    def test_complex_default_and_legacy_trapezoids(self):
        values = np.full(4, 1j)
        for args in ((), ({0: 'trapezoidal'},), ([0.5],)):
            with self.subTest(args=args):
                self.assertEqual(int_arr(values, *args), 3j)

    def test_full_rules_preserve_complex_polynomial_on_physical_interval(self):
        for name, nodes in [('chebyshev-gauss', mapped_chebyshev_gauss_nodes),
                            ('clenshaw-curtis', mapped_chebyshev_nodes)]:
            with self.subTest(name=name):
                x = nodes(5, 0, 2)
                values = x**2 + 1j * x
                result = int_arr(values, {0: {name: {'nodes': x, 'interval': (0, 2)}}})
                np.testing.assert_allclose(result, 8 / 3 + 2j)

    def test_truncated_lobatto_uses_parent_node_count(self):
        x = truncated_chebyshev_nodes(9, -0.5, 0.5)
        self.assertEqual(len(x), 3)
        rule = {0: {'truncated-cc-grid': {'num_nodes': 9, 'interval': (-0.5, 0.5)}}}
        np.testing.assert_allclose(int_arr(x**2, rule), 1 / 12)

    def test_degree_is_forwarded_in_all_moment_rules(self):
        a, b = -0.5, 0.75
        cases = [('truncated-cc-grid', truncated_chebyshev_nodes(13, a, b)),
                 ('truncated-cg-grid', truncated_chebyshev_gauss_nodes(13, a, b)),
                 ('moment-matching', np.array([a, -0.25, 0.1, b]))]
        for name, x in cases:
            # Degree zero gives equal weights summing to the interval length.
            values = x**2
            expected = (b - a) * np.mean(values)
            self.assertFalse(np.isclose(expected, (b**3 - a**3) / 3))
            for local in (False, True):
                with self.subTest(name=name, local=local):
                    opts = {'nodes': x, 'interval': (a, b), 'degree': 0}
                    np.testing.assert_allclose(int_arr(values, {0: {name: opts}},
                                                       domain_local_weight=local),
                                               expected)

    def test_weight_scale_applied_once_over_two_axes(self):
        w = Weight(m=[2, 2], q=[0, 0], k=[0, 0], dxs=[1, 1], scale=3.)
        x = mapped_chebyshev_nodes(5, -1, 1)
        for weights in (clenshaw_curtis_weights(5, weight=w),
                        chebyshev_gauss_weights(5, weight=w, axis=1),
                        moment_matched_quad_weights(x, -1, 1, weight=w)):
            np.testing.assert_allclose(weights.sum(), 16 / 15)
        result = int_arr(np.ones((5, 5)), {0: 'clenshaw-curtis', 1: 'chebyshev-gauss'}, weight=w)
        np.testing.assert_allclose(result, 3 * (16 / 15)**2)

    def test_dataset_integrates_a_physical_slice_and_uniform_time_steps(self):
        x = mapped_chebyshev_nodes(7, 1., 4.)
        dt = 0.25
        p = Observable(string='p', rank=0, indices=())
        values = np.broadcast_to(x[:, None]**2, (7, 4))
        ds = SRDataset(world_size=[7, 4], data_dict={'p': values}, observables=[p],
                       cache_primes=False, dxs=[1., dt], grids=[x, None],
                       schemes_and_options={0: 'truncated-cc-grid'})
        window = IntegrationDomain([1, 1], [5, 3])
        prime = LibraryPrime(derivand=p, derivative=DerivativeOrder(x_derivatives=()))
        term = LibraryTerm(primes=(prime,), rank=0)
        weight = Weight(m=[0, 0], q=[0, 0], k=[0, 0], dxs=[1., 1.])
        raw = ds.eval_on_domain(term, weight, window)
        expected = (x[5]**3 - x[1]**3) / 3 * (2 * dt)
        np.testing.assert_allclose(raw * dt, expected)
