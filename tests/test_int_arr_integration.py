"""Tests for int_arr Chebyshev scheme dispatch in process_library_terms."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySPIDER.commons.chebyshev_integration import (  # noqa: E402
    chebyshev_lobatto_nodes,
    integrate as cheb_integrate,
    mapped_chebyshev_nodes,
    truncated_chebyshev_nodes,
)
from PySPIDER.commons.process_library_terms import int_arr  # noqa: E402


def test_int_arr_default_trapz_matches_numpy():
    vals = np.sin(np.linspace(-1.0, 1.0, 33))
    assert np.isclose(int_arr(vals), np.trapezoid(vals), atol=1e-14)


def test_int_arr_clenshaw_curtis_full_interval():
    N = 64
    nodes = chebyshev_lobatto_nodes(N)
    f = lambda x: np.exp(-x) * np.cos(6.0 * x)
    vals = f(nodes)
    ref, _ = quad(f, -1.0, 1.0, limit=200)
    direct = cheb_integrate(nodes, vals, -1.0, 1.0)
    via_int_arr = int_arr(vals, scheme="clenshaw-curtis")
    assert abs(direct - ref) < 1e-12 * (1 + abs(ref))
    assert abs(via_int_arr - direct) < 1e-14 * (1 + abs(direct))


def test_int_arr_truncated_grid_subinterval():
    a, b = -0.4, 0.7
    nodes = truncated_chebyshev_nodes(96, a, b)
    vals = np.exp(nodes)
    ref = np.exp(b) - np.exp(a)
    direct = cheb_integrate(nodes, vals, a, b)
    via_int_arr = int_arr(
        vals,
        scheme="truncated-grid",
        scheme_options={"nodes": nodes, "interval": (a, b)},
    )
    assert abs(direct - ref) < 1e-10 * (1 + abs(ref))
    assert abs(via_int_arr - direct) < 1e-14 * (1 + abs(direct))


def test_int_arr_truncated_grid_jacobi_weight():
    nodes = truncated_chebyshev_nodes(96, -1.0, 1.0)
    m = 3.0
    ref, _ = quad(lambda x: (1.0 - x**2) ** m * np.cos(x), -1.0, 1.0, limit=200)
    vals = np.cos(nodes)
    actual = int_arr(
        vals,
        scheme="truncated_grid",
        scheme_options={"nodes": nodes, "interval": (-1.0, 1.0), "m": m},
    )
    assert abs(actual - ref) < 1e-9 * (1 + abs(ref))


def test_int_arr_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="Unsupported integration scheme"):
        int_arr(np.ones(5), scheme="simpson")


def test_int_arr_truncated_requires_num_intervals_off_reference():
    vals = np.ones(11)
    with pytest.raises(ValueError, match="num_intervals"):
        int_arr(
            vals,
            scheme="truncated-grid",
            scheme_options={"interval": (0.0, 1.0)},
        )


def test_int_arr_clenshaw_curtis_affine_interval():
    a, b = 0.0, 2.0
    N = 48
    nodes = mapped_chebyshev_nodes(N, a, b)
    f = lambda x: np.exp(-x) * np.cos(4.0 * x)
    vals = f(nodes)
    ref, _ = quad(f, a, b, limit=200)
    direct = cheb_integrate(nodes, vals, a, b)
    via_int_arr = int_arr(
        vals,
        scheme="clenshaw-curtis",
        scheme_options={"interval": (a, b), "nodes": nodes},
    )
    assert abs(direct - ref) < 1e-12 * (1 + abs(ref))
    assert abs(via_int_arr - direct) < 1e-14 * (1 + abs(direct))


def test_int_arr_clenshaw_curtis_affine_interval_default_nodes():
    a, b = -0.2, 0.9
    N = 32
    nodes = mapped_chebyshev_nodes(N, a, b)
    vals = np.sin(2.0 * nodes)
    ref, _ = quad(lambda x: np.sin(2.0 * x), a, b, limit=200)
    via_int_arr = int_arr(
        vals,
        scheme="clenshaw-curtis",
        scheme_options={"interval": (a, b)},
    )
    assert abs(via_int_arr - ref) < 1e-12 * (1 + abs(ref))
