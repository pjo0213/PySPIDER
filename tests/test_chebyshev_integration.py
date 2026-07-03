"""Unit tests for PySPIDER.commons.chebyshev_integration."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySPIDER.commons.chebyshev_integration import (  # noqa: E402
    chebyshev_lobatto_nodes,
    chebyshev_coefficients_from_values,
    clenshaw_curtis_weights,
    clenshaw_curtis_weights_on_interval,
    integrate,
    integrate_clenshaw_curtis_on_interval,
    integrate_weighted_clenshaw_curtis_from_values,
    jacobi_weighted_moments_T,
    jacobi_weighted_moments_T_dct,
    mapped_chebyshev_nodes,
    truncated_chebyshev_nodes,
)


def test_clenshaw_curtis_weights_match_dct_integrator():
    N = 48
    f = lambda x: np.exp(-x) * np.cos(5.0 * x)
    nodes = chebyshev_lobatto_nodes(N)
    vals = f(nodes)
    w = clenshaw_curtis_weights(N)
    ref, _ = quad(f, -1.0, 1.0, limit=200)
    assert abs(np.dot(w, vals) - ref) < 1e-12 * (1 + abs(ref))
    assert abs(integrate_weighted_clenshaw_curtis_from_values(nodes, vals, 0.0) - ref) < 1e-12 * (
        1 + abs(ref)
    )


def test_ascending_lobatto_order_matches_descending():
    N = 32
    f = lambda x: np.sin(3.0 * x)
    nodes_desc = chebyshev_lobatto_nodes(N)
    nodes_asc = np.sort(nodes_desc)
    ref, _ = quad(f, -1.0, 1.0, limit=200)
    err_desc = abs(integrate_weighted_clenshaw_curtis_from_values(nodes_desc, f(nodes_desc), 0.0) - ref)
    err_asc = abs(integrate_weighted_clenshaw_curtis_from_values(nodes_asc, f(nodes_asc), 0.0) - ref)
    assert err_desc < 1e-12
    assert err_asc < 1e-12


def test_jacobi_moments_dct_vs_quadpack():
    m = 2.5
    max_degree = 24
    mu_dct = jacobi_weighted_moments_T_dct(m, max_degree)
    mu_quad = jacobi_weighted_moments_T(m, max_degree)
    assert np.allclose(mu_dct, mu_quad, rtol=0, atol=1e-10)


def test_weighted_integral():
    m = 2.0
    N = 64
    nodes = chebyshev_lobatto_nodes(N)
    f = lambda x: np.cos(4.0 * x)
    ref, _ = quad(lambda x: (1.0 - x**2) ** m * f(x), -1.0, 1.0, limit=200)
    val = integrate_weighted_clenshaw_curtis_from_values(nodes, f(nodes), m)
    assert abs(val - ref) < 1e-11 * (1 + abs(ref))


def test_truncated_subinterval_lstsq():
    a, b = -0.3, 0.7
    nodes = truncated_chebyshev_nodes(128, a, b)
    ref, _ = quad(lambda x: np.exp(x), a, b, limit=200)
    val = integrate(nodes, np.exp(nodes), a, b)
    assert abs(val - ref) < 1e-11 * (1 + abs(ref))


def test_affine_clenshaw_curtis_on_interval():
    a, b = 0.0, 4.0
    N = 48
    nodes = mapped_chebyshev_nodes(N, a, b)
    f = lambda x: np.exp(-0.5 * x) * np.sin(3.0 * x)
    vals = f(nodes)
    ref, _ = quad(f, a, b, limit=200)
    direct = integrate_clenshaw_curtis_on_interval(a, b, vals, nodes=nodes)
    via_integrate = integrate(nodes, vals, a, b)
    assert abs(direct - ref) < 1e-12 * (1 + abs(ref))
    assert abs(via_integrate - direct) < 1e-14 * (1 + abs(direct))


def test_affine_cc_weights_match_interval_integrator():
    a, b = -0.5, 2.5
    N = 32
    nodes = mapped_chebyshev_nodes(N, a, b)
    f = lambda x: np.cos(2.0 * x)
    vals = f(nodes)
    w = clenshaw_curtis_weights_on_interval(a, b, N)
    ref, _ = quad(f, a, b, limit=200)
    assert abs(np.dot(w, vals) - ref) < 1e-12 * (1 + abs(ref))


def test_affine_envelope_weighted_cc():
    a, b = 0.0, 1.0
    m = 2.0
    N = 40
    nodes = mapped_chebyshev_nodes(N, a, b)
    f = lambda x: np.cos(5.0 * x)
    ref, _ = quad(lambda x: ((x - a) * (b - x)) ** m * f(x), a, b, limit=200)
    val = integrate_clenshaw_curtis_on_interval(
        a, b, f(nodes), nodes=nodes, envelope_m=m
    )
    assert abs(val - ref) < 1e-11 * (1 + abs(ref))


def test_affine_cc_matches_quad_reference():
    """Affine DCT Clenshaw-Curtis should match high-precision quadrature."""
    a, b = 0.2, 1.8
    N = 24
    nodes = mapped_chebyshev_nodes(N, a, b)
    f = lambda x: np.exp(-x) * np.cos(3.0 * x)
    vals = f(nodes)
    dct_val = integrate(nodes, vals, a, b)
    ref, _ = quad(f, a, b, limit=200)
    assert abs(dct_val - ref) < 1e-12 * (1 + abs(ref))


def test_truncated_nodes_rejects_empty_interval():
    with pytest.raises(ValueError, match="No Chebyshev-Lobatto nodes"):
        truncated_chebyshev_nodes(4, 0.9, 0.91)


def test_integrate_rejects_non_lobatto_grid_on_reference_interval():
    N = 16
    lobatto = chebyshev_lobatto_nodes(N)
    uniform = np.linspace(-1.0, 1.0, N + 1)
    with pytest.raises(ValueError, match="full Chebyshev-Lobatto"):
        integrate_weighted_clenshaw_curtis_from_values(uniform, np.ones_like(uniform), 0.0)


def test_chebyshev_coefficients_reconstruct_polynomial():
    N = 12
    nodes = chebyshev_lobatto_nodes(N)
    # f(x) = 1 + 2 T_1(x) + 0.5 T_3(x)
    f_vals = 1.0 + 2.0 * nodes + 0.5 * (4.0 * nodes**3 - 3.0 * nodes)
    coeffs = chebyshev_coefficients_from_values(f_vals)
    assert abs(coeffs[0] - 1.0) < 1e-12
    assert abs(coeffs[1] - 2.0) < 1e-12
    assert abs(coeffs[3] - 0.5) < 1e-12
    assert np.max(np.abs(coeffs[[2, 4, 5, 6, 7, 8, 9, 10, 11, 12]])) < 1e-11
