"""Quadrature weights, nodes, and polynomial exactness (Gauss and Lobatto)."""

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebpts1
from scipy.fft import dct

from PySPIDER.commons.quadrature_schemes import (
    chebyshev_gauss_weights,
    clenshaw_curtis_weights,
    mapped_chebyshev_gauss_nodes,
    mapped_chebyshev_nodes,
    moment_matched_quad_weights,
    truncated_chebyshev_gauss_nodes,
    _fejer_weights_from_moments,
    _compute_moments,
    _mapped_gauss_reference_degree,
    _mapped_lobatto_reference_degree,
)


@pytest.mark.parametrize("n", [2, 3, 4, 7, 8, 16, 32])
def test_gauss_dct_iii_weights_match_moment_matching(n):
    """Fejér-I / DCT-III weights agree with interpolatory moment-matching."""
    nodes = mapped_chebyshev_gauss_nodes(n, -1.0, 1.0)
    w_dct = chebyshev_gauss_weights(n, -1.0, 1.0)
    w_mm = moment_matched_quad_weights(nodes, -1.0, 1.0)
    np.testing.assert_allclose(w_dct, w_mm, rtol=0, atol=1e-12)


@pytest.mark.parametrize("num_intervals", [2, 4, 8, 16])
def test_lobatto_clenshaw_curtis_weights_match_moment_matching(num_intervals):
    nodes = mapped_chebyshev_nodes(num_intervals, -1.0, 1.0)
    w_cc = clenshaw_curtis_weights(num_intervals, -1.0, 1.0)
    w_mm = moment_matched_quad_weights(nodes, -1.0, 1.0)
    np.testing.assert_allclose(w_cc, w_mm, rtol=0, atol=1e-12)


def test_fejer_helper_is_scipy_dct_type_3():
    n = 10
    mu = _compute_moments(-1.0, 1.0, n - 1)
    np.testing.assert_allclose(
        _fejer_weights_from_moments(mu),
        (dct(np.asarray(mu, dtype=float), type=3) / n)[::-1],
        rtol=0,
        atol=1e-15,
    )


def test_gauss_nodes_match_sorted_dedalus_roots():
    n = 8
    dedalus_desc = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    ours = mapped_chebyshev_gauss_nodes(n, -1.0, 1.0)
    np.testing.assert_allclose(ours, np.sort(dedalus_desc))
    np.testing.assert_allclose(ours, chebpts1(n))
    assert np.abs(ours[0] + 1.0) > 1e-6
    assert np.abs(ours[-1] - 1.0) > 1e-6


@pytest.mark.parametrize("n", [4, 8, 16])
def test_gauss_polynomial_exactness_on_minus1_1(n):
    """n Gauss nodes integrate polynomials of degree < n exactly on [-1, 1]."""
    x = mapped_chebyshev_gauss_nodes(n, -1.0, 1.0)
    w = chebyshev_gauss_weights(n, -1.0, 1.0)
    for p in range(n):
        exact = 2.0 / (p + 1) if p % 2 == 0 else 0.0
        approx = float(np.dot(w, x**p))
        assert abs(approx - exact) <= 1e-12 * max(1.0, abs(exact))


@pytest.mark.parametrize("num_intervals", [4, 8, 16])
def test_lobatto_polynomial_exactness_on_minus1_1(num_intervals):
    """N+1 Lobatto nodes integrate polynomials of degree <= N exactly on [-1, 1]."""
    x = mapped_chebyshev_nodes(num_intervals, -1.0, 1.0)
    w = clenshaw_curtis_weights(num_intervals, -1.0, 1.0)
    for p in range(num_intervals + 1):
        exact = 2.0 / (p + 1) if p % 2 == 0 else 0.0
        approx = float(np.dot(w, x**p))
        assert abs(approx - exact) <= 1e-12 * max(1.0, abs(exact))


def test_mapped_reference_degree_accepts_ascending_and_descending():
    gauss = mapped_chebyshev_gauss_nodes(9, 0.0, 1.0)
    lobatto = mapped_chebyshev_nodes(8, 0.0, 1.0)
    assert _mapped_gauss_reference_degree(gauss, 0.0, 1.0) == 9
    assert _mapped_gauss_reference_degree(gauss[::-1], 0.0, 1.0) == 9
    assert _mapped_lobatto_reference_degree(lobatto, 0.0, 1.0) == 8
    assert _mapped_lobatto_reference_degree(lobatto[::-1], 0.0, 1.0) == 8
    assert _mapped_gauss_reference_degree(lobatto, 0.0, 1.0) is None
    assert _mapped_lobatto_reference_degree(gauss, 0.0, 1.0) is None


def test_truncated_gauss_nodes_are_a_subset_of_the_parent_grid():
    n_parent = 32
    a, b = -0.4, 0.7
    full = mapped_chebyshev_gauss_nodes(n_parent, -1.0, 1.0)
    sub = truncated_chebyshev_gauss_nodes(n_parent, a, b)
    assert sub.size >= 2
    assert np.all(sub >= a - 1e-12)
    assert np.all(sub <= b + 1e-12)
    # Every kept node is a parent-grid node (within float noise).
    for x in sub:
        assert np.min(np.abs(full - x)) < 1e-12
