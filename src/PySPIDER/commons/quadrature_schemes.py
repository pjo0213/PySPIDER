# All implemented quadrature schemes

from typing import Optional

import numpy as np
from numpy.polynomial.chebyshev import chebint, chebpts2, chebval, chebvander, poly2cheb
from scipy.fft import dct


#helper function used quite a bit
def _validate_interval(a: float, b: float) -> None:
    if a > b:
        raise ValueError(f"interval endpoints must satisfy a <= b, got ({a}, {b})")

def mapped_chebyshev_nodes(N: int, a: float, b: float) -> np.ndarray:
    """
    Affinely map the Chebyshev-Lobatto nodes from [-1, 1] onto [a, b].

    Uses `numpy.polynomial.chebyshev.chebpts2` for the reference nodes, then
    applies x -> (a+b)/2 + (b-a)/2 * x and returns them sorted ascending.

    Parameters
    ----------
    N : int
        Number of intervals; produces N+1 nodes.
    a : float
        Left endpoint of the target interval.
    b : float
        Right endpoint of the target interval.

    Returns
    -------
    np.ndarray
        Array of N+1 nodes mapped onto [a, b].
    """
    _validate_interval(a, b)
    x = chebpts2(N + 1)
    return (0.5 * (a + b) + 0.5 * (b - a) * x)

def truncated_chebyshev_nodes(N: int, a: float, b: float) -> np.ndarray:
    """
    Return the Chebyshev-Lobatto nodes that fall within [a, b].

    The full reference grid is generated with
    `numpy.polynomial.chebyshev.chebpts2` and only the subset lying in
    [a, b] (within a floating-point tolerance) is retained, sorted ascending.

    Parameters
    ----------
    N : int
        Number of intervals of the underlying reference grid.
    a : float
        Left endpoint of the retained interval.
    b : float
        Right endpoint of the retained interval.

    Returns
    -------
    np.ndarray
        Array of the reference nodes contained in [a, b].
    """
    _validate_interval(a, b)
    nodes = chebpts2(N + 1)
    tol = np.finfo(float).eps * max(abs(a), abs(b), 1.0)
    mask = (nodes >= a - tol) & (nodes <= b + tol)
    kept = nodes[mask]
    if kept.size == 0:
        raise ValueError(
            f"No Chebyshev-Lobatto nodes from reference grid N = {N} lie in [{a}, {b}]. "
            "Increase num_intervals or widen the interval."
        )
    return kept

def _chebyshev_coefficients_from_values(f_values: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev coefficients from Lobatto-grid samples via a DCT-I.

    Given samples at the Chebyshev-Lobatto nodes, returns the coefficients
    a_k of the expansion f(x) ≈ sum_{k=0}^{N} a_k T_k(x).

    Parameters
    ----------
    f_values : np.ndarray
        Samples of f at the N+1 Lobatto nodes.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients a_0, ..., a_N.
    """
    N = f_values.shape[0] - 1
    if N < 1:
        raise ValueError("At least two sample points (N>=1) are required")
    a = dct(np.asarray(f_values, dtype=float), type=1) / N
    a[0] *= 0.5
    a[-1] *= 0.5
    return a

def _ensure_weight_ready(weight) -> None:
    if not getattr(weight, "ready", False) or weight.weight_objs is None:
        weight.make_weight_objs()

def _polynomial_weighted_moments(poly, a: float, b: float, max_degree: int) -> np.ndarray:
    """
    Exact Chebyshev moments of a polynomial weight.

    Returns mu_j = integral_a^b W(x) T_j(x) dx. W is expanded as a
    Chebyshev series and each product T_i T_j is integrated with
    ``_unweighted_moments``.
    """
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")
    coef = np.asarray(poly.coef, dtype=float).ravel()
    if coef.size == 0 or not np.any(coef):
        return np.zeros(max_degree + 1)
    a_w = poly2cheb(coef)
    I = _unweighted_moments(a, b, a_w.shape[0] - 1 + max_degree)
    mu = np.zeros(max_degree + 1)
    js = np.arange(max_degree + 1)
    for i, ai in enumerate(a_w):
        if ai == 0.0:
            continue
        mu += 0.5 * ai * (I[i + js] + I[np.abs(i - js)])
    return mu

def _unweighted_moments(a: float, b: float, max_degree: int) -> np.ndarray:
    """
    Compute the unweighted Chebyshev moments analytically.

    Returns mu_k = integral_a^b T_k(x) dx by integrating the Chebyshev basis
    with `numpy.polynomial.chebyshev.chebint` (antiderivative vanishing at a)
    and evaluating at b.

    Parameters
    ----------
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    max_degree : int
        Largest Chebyshev index k.

    Returns
    -------
    np.ndarray
        Array of moments mu_0, ..., mu_max_degree.
    """
    return chebval(b, chebint(np.eye(max_degree + 1), lbnd=a, axis=0))

#used in integration.py
def _mapped_lobatto_reference_degree(nodes: np.ndarray, a: float, b: float) -> Optional[int]:
    """
    Return N when nodes are a full Lobatto grid affinely mapped onto [a, b].

    Accepts ascending or descending node orderings.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size < 2:
        return None
    _validate_interval(a, b)
    N = nodes.shape[0] - 1
    expected = mapped_chebyshev_nodes(N, a, b)
    if np.allclose(nodes, expected, rtol=0, atol=1e-12):
        return N
    if np.allclose(nodes, expected[::-1], rtol=0, atol=1e-12):
        return N
    return None

def _clenshaw_curtis_from_moments(mu: np.ndarray) -> np.ndarray:
    """Clenshaw-Curtis node weights from Chebyshev moments via DCT-I.

    DCT-I emits weights for ``x_j = cos(pi j / N)`` (1 -> -1). Reverse so
    they are ascending.
    """
    return _chebyshev_coefficients_from_values(mu)[::-1]

def clenshaw_curtis_weights(
    num_intervals: int,
    a: float = -1.0,
    b: float = 1.0,
    *,
    weight=None,
    axis: int = 0,
) -> np.ndarray:
    """
    Clenshaw-Curtis weights on the full Lobatto grid mapped to [a, b].

    Returns weights w_j such that sum_j w_j f(x_j) approximates

        integral_a^b w(s(x)) f(x) dx,

    where s maps [a, b] onto [-1, 1]. ``w`` is 1 or one axis of a
    PySPIDER ``Weight``. Uses reference Clenshaw-Curtis weights times
    the affine Jacobian (b-a)/2. Weights are in ascending order.

    Parameters
    ----------
    num_intervals : int
        Number of grid intervals N; the grid has N+1 nodes.
    a, b : float, optional
        Interval endpoints; default [-1, 1].
    weight : optional
        PySPIDER Weight; moments use that object's attributes for ``axis``.
    axis : int, optional
        Axis of ``weight`` to use.

    Returns
    -------
    np.ndarray
        Array of N+1 quadrature weights in ascending node order.
    """
    _validate_interval(a, b)
    N = num_intervals
    if N < 1:
        raise ValueError("num_intervals must be >= 1")
    mu = _compute_moments(-1.0, 1.0, N, weight=weight, axis=axis)
    return 0.5 * (b - a) * _clenshaw_curtis_from_moments(mu)

#moment matching and relevant helper functions
def _compute_moments(
    a: float,
    b: float,
    max_degree: int,
    *,
    weight=None,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute Chebyshev moments for a chosen weight on [a, b].

    Returns mu_k = integral_a^b w(x) T_k(x) dx. A PySPIDER ``weight``
    is parsed for axis ``axis`` (the 1-D polynomial on that axis,
    including m, q, k, dx). With no weight, w(x) = 1. Both cases are
    evaluated from Chebyshev antiderivatives of the polynomial integrand.

    Parameters
    ----------
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    max_degree : int
        Largest Chebyshev index k.
    weight : optional
        PySPIDER Weight; moments use that object's attributes for ``axis``.
    axis : int, optional
        Axis of ``weight`` to use.

    Returns
    -------
    np.ndarray
        Array of moments mu_0, ..., mu_max_degree.
    """
    if weight is not None:
        _ensure_weight_ready(weight)
        return _polynomial_weighted_moments(
            weight.weight_objs[axis], a, b, max_degree
        )
    return _unweighted_moments(a, b, max_degree)

def moment_matched_quad_weights(
    nodes: np.ndarray,
    a: float,
    b: float,
    *,
    weight=None,
    axis: int = 0,
    degree: Optional[int] = None,
) -> np.ndarray:
    """
    Compute quadrature weights for arbitrary nodes by moment matching.

    Solves the (least-squares) moment-matching system V^T w = mu, where V is
    the Chebyshev basis matrix at nodes (`numpy.polynomial.chebyshev.chebvander`)
    and mu are the weighted moments on [a, b].
    
    Parameters
    ----------
    nodes : np.ndarray
        Quadrature nodes in [a, b].
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    weight : optional
        PySPIDER Weight; moments use that object's attributes for ``axis``.
    axis : int, optional
        Axis of ``weight`` to use.
    degree : int, optional
        Polynomial degree to match; defaults to len(nodes) - 1.

    Returns
    -------
    np.ndarray
        Quadrature weights aligned with nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    _validate_interval(a, b)
    M = len(nodes)
    if M < 2:
        raise ValueError("moment_matched_quad_weights requires at least two nodes (degree >= 1).")

    if degree is None:
        degree = M - 1

    V = chebvander(nodes, degree)
    mu = _compute_moments(a, b, degree, weight=weight, axis=axis)
    return np.linalg.lstsq(V.T, mu, rcond=None)[0]