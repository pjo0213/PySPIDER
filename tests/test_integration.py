"""int_arr scheme selection: Gauss vs Lobatto, Dedalus order, mismatch, convergence."""

import numpy as np
import pytest
from scipy.integrate import quad

from PySPIDER.commons.integration import SUPPORTED_SCHEMES, int_arr
from PySPIDER.commons.quadrature_schemes import (
    mapped_chebyshev_gauss_nodes,
    mapped_chebyshev_nodes,
    truncated_chebyshev_gauss_nodes,
)

# Smooth integrand used in the PR checks (not a polynomial, so truncation error
# is visible when the wrong Chebyshev grid is assumed).
_F = lambda x: np.exp(-x) * np.cos(3.0 * x)
_F_CONV = lambda x: np.exp(-4.0 * x) * np.cos(12.0 * x)


def test_supported_schemes_include_gauss_and_lobatto():
    assert "clenshaw-curtis" in SUPPORTED_SCHEMES
    assert "chebyshev-gauss" in SUPPORTED_SCHEMES
    assert "truncated-cc-grid" in SUPPORTED_SCHEMES
    assert "truncated-cg-grid" in SUPPORTED_SCHEMES


def test_int_arr_chebyshev_gauss_agrees_with_quad():
    a, b = 0.0, 2.0
    n = 12
    x = mapped_chebyshev_gauss_nodes(n, a, b)
    ref, _ = quad(_F, a, b)
    val = float(int_arr(_F(x), {0: {"chebyshev-gauss": {"interval": (a, b)}}}))
    assert abs(val - ref) < 1e-8


def test_int_arr_chebyshev_gauss_descending_dedalus_order():
    a, b = 0.0, 2.0
    n = 12
    x_asc = mapped_chebyshev_gauss_nodes(n, a, b)
    x_desc = x_asc[::-1]
    ref, _ = quad(_F, a, b)
    val = float(
        int_arr(
            _F(x_desc),
            {0: {"chebyshev-gauss": {"interval": (a, b), "nodes": x_desc}}},
        )
    )
    assert abs(val - ref) < 1e-8


def test_int_arr_clenshaw_curtis_agrees_with_quad():
    a, b = 0.0, 2.0
    n = 12
    x = mapped_chebyshev_nodes(n - 1, a, b)
    ref, _ = quad(_F, a, b)
    val = float(int_arr(_F(x), {0: {"clenshaw-curtis": {"interval": (a, b)}}}))
    assert abs(val - ref) < 1e-8


def test_int_arr_truncated_cg_grid_agrees_with_quad():
    a, b = -0.4, 0.7
    n_parent = 32
    x = truncated_chebyshev_gauss_nodes(n_parent, a, b)
    ref, _ = quad(_F, a, b)
    val = float(
        int_arr(
            _F(x),
            {0: {"truncated-cg-grid": {"interval": (a, b), "num_nodes": n_parent}}},
        )
    )
    assert abs(val - ref) < 1e-8


def test_clenshaw_curtis_rejects_gauss_nodes():
    a, b = 0.0, 2.0
    xg = mapped_chebyshev_gauss_nodes(12, a, b)
    with pytest.raises(ValueError, match="Lobatto"):
        int_arr(
            _F(xg),
            {0: {"clenshaw-curtis": {"interval": (a, b), "nodes": xg}}},
        )


def test_chebyshev_gauss_rejects_lobatto_nodes():
    a, b = 0.0, 2.0
    xl = mapped_chebyshev_nodes(11, a, b)
    with pytest.raises(ValueError, match="Gauss"):
        int_arr(
            _F(xl),
            {0: {"chebyshev-gauss": {"interval": (a, b), "nodes": xl}}},
        )


def test_convergence_correct_grids_reach_machine_precision_wrong_grid_stalls():
    """Artem failure mode: Clenshaw–Curtis inferred on Gauss samples does not converge.

    Correct Gauss and Lobatto rules on their own nodes go to ~machine precision.
    Inferring a Lobatto grid for Gauss samples (no ``nodes`` passed) stalls
    around 1e-2.
    """
    a, b = 0.0, 1.0
    ref, _ = quad(_F_CONV, a, b)
    n = 32

    xg = mapped_chebyshev_gauss_nodes(n, a, b)
    err_gauss = abs(
        float(int_arr(_F_CONV(xg), {0: {"chebyshev-gauss": {"interval": (a, b)}}}))
        - ref
    )
    xl = mapped_chebyshev_nodes(n - 1, a, b)
    err_lobatto = abs(
        float(int_arr(_F_CONV(xl), {0: {"clenshaw-curtis": {"interval": (a, b)}}}))
        - ref
    )
    # Infer Lobatto nodes from array length while samples live at Gauss nodes.
    err_wrong = abs(
        float(int_arr(_F_CONV(xg), {0: {"clenshaw-curtis": {"interval": (a, b)}}}))
        - ref
    )

    assert err_gauss < 1e-12
    assert err_lobatto < 1e-12
    assert err_wrong > 1e-3
    assert err_wrong > 1e6 * max(err_gauss, np.finfo(float).eps)
