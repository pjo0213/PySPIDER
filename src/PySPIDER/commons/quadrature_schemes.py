# All implemented quadrature schemes

from typing import Callable, Optional

import numpy as np
from numpy.polynomial.chebyshev import chebint, chebpts2, chebval, chebvander
from scipy.fft import dct
from scipy.integrate import quad


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

def chebyshev_coefficients_from_values(f_values: np.ndarray) -> np.ndarray:
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

def jacobi_weighted_moments_T_dct(m: float, max_degree: int) -> np.ndarray:
    """
    Compute Jacobi-weighted Chebyshev moments via a DCT.

    Returns the moments mu_k = integral_{-1}^{1} (1-x^2)^m T_k(x) dx. The
    substitution x = cos(theta) turns this into integral_0^pi
    sin^{2m+1}(theta) cos(k theta) dtheta. A DCT-I of sin^{2m+1} yields
    cosine coefficients c_k; the moments are then mu_0 = pi * c_0 and
    mu_k = (pi/2) * c_k for k >= 1.

    Assumes m >= 2.

    Parameters
    ----------
    m : float
        Weight exponent in (1-x^2)^m; must be >= 2.
    max_degree : int
        Largest Chebyshev index k.

    Returns
    -------
    np.ndarray
        Array of moments mu_0, ..., mu_max_degree.
    """
    if m < 2:
        raise ValueError("m must be >= 2 for DCT-based moments.")
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")
    M = 2 * max_degree # since weight and function domains are decoupled, change if necessary
    theta = np.pi * np.arange(M + 1, dtype=float) / M
    f_theta = np.sin(theta) ** (2.0 * m + 1.0)
    c = chebyshev_coefficients_from_values(f_theta)
    mu = 0.5 * np.pi * c[: max_degree + 1]
    mu[0] = np.pi * c[0]
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

def _jacobi_weight_on_interval(a: float, b: float, m: float) -> Callable[[float], float]:
    """Return (1-s(x)^2)^m, where s maps [a, b] onto [-1, 1]."""
    scale = 4.0 / (b - a) ** 2

    def weight(x, m=m, a=a, b=b, scale=scale):
        return (scale * (x - a) * (b - x)) ** m

    return weight


def clenshaw_curtis_weights(
    num_intervals: int,
    m: float = 0.0,
) -> np.ndarray:
    """
    Compute Clenshaw-Curtis quadrature weights on the full Lobatto grid.

    Returns weights w_j such that sum_j w_j f(x_j) = integral_{-1}^{1}
    (1-x^2)^m f(x) dx for f sampled at the Lobatto nodes x_j = cos(pi*j/N),
    j = 0..N.

    Parameters
    ----------
    num_intervals : int
        Number of grid intervals N; the grid has N+1 nodes.
    m : float, optional
        Weight exponent in (1-x^2)^m.

    Returns
    -------
    np.ndarray
        Array of N+1 quadrature weights.
    """
    N = num_intervals
    if N < 1:
        raise ValueError("num_intervals must be >= 1")
    if m == 0:
        mu = _unweighted_moments(-1.0, 1.0, N)
    else:
        mu = jacobi_weighted_moments_T_dct(m, N)
    j = np.arange(N + 1)
    C = np.cos(np.pi * np.outer(j, j) / N) / N
    C[:, 1:-1] *= 2.0
    C[0, :] *= 0.5
    C[-1, :] *= 0.5
    w = C @ mu
    return w

def clenshaw_curtis_weights_on_interval(
    a: float,
    b: float,
    num_intervals: int,
    m: float = 0.0,
) -> np.ndarray:
    """
    Clenshaw-Curtis weights on the full Lobatto grid mapped to [a, b].

    Returns weights w_j such that sum_j w_j f(x_j) approximates

        integral_a^b (1-s(x)^2)^m f(x) dx,

    where s maps [a, b] onto [-1, 1] (so m=0 is the unweighted integral).
    Uses the reference Clenshaw-Curtis weights times the affine Jacobian
    (b-a)/2. CC weights are symmetric, so ascending or descending sample
    order is equivalent.
    """
    _validate_interval(a, b)
    return 0.5 * (b - a) * clenshaw_curtis_weights(num_intervals, m=m)

#moment matching and relevant helper functions
def _moments_via_quad(
    a: float, b: float, max_degree: int, w_fn: Callable[[float], float]
) -> np.ndarray:
    """
    Compute general weighted Chebyshev moments via numerical quadrature.

    Returns mu_k = integral_a^b w(x) T_k(x) dx using scipy.integrate.quad for
    an arbitrary weight w.

    Parameters
    ----------
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    max_degree : int
        Largest Chebyshev index k.
    w_fn : callable
        Weight function w(x).

    Returns
    -------
    np.ndarray
        Array of moments mu_0, ..., mu_max_degree.
    """
    mu = np.zeros(max_degree + 1)
    for k in range(max_degree + 1):
        def integrand(x, kk=k):
            c = np.zeros(kk + 1)
            c[kk] = 1.0
            return w_fn(x) * chebval(x, c)

        val, _ = quad(integrand, a, b, limit=200, epsabs=1e-13, epsrel=1e-13)
        mu[k] = val
    return mu

def compute_moments(
    a: float,
    b: float,
    max_degree: int,
    *,
    weight_func: Optional[Callable] = None,
    m: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Chebyshev moments for a chosen weight on [a, b].

    Returns mu_k = integral_a^b w(x) T_k(x) dx. At most one of weight_func
    or m may be given. m is the Jacobi exponent (1-s(x)^2)^m, where s maps
    [a, b] onto [-1, 1]. With neither, w(x) = 1.

    Parameters
    ----------
    a : float
        Left endpoint of integration.
    b : float
        Right endpoint of integration.
    max_degree : int
        Largest Chebyshev index k.
    weight_func : callable, optional
        Arbitrary weight function w(x).
    m : float, optional
        Exponent of the Jacobi weight (1-s(x)^2)^m.

    Returns
    -------
    np.ndarray
        Array of moments mu_0, ..., mu_max_degree.
    """
    if weight_func is not None and m is not None:
        raise ValueError("Specify at most one of weight_func, m")

    if weight_func is None and (m is None or m == 0):
        return _unweighted_moments(a, b, max_degree)

    if m is not None:
        if np.isclose(a, -1.0) and np.isclose(b, 1.0) and m >= 2:
            return jacobi_weighted_moments_T_dct(m, max_degree)
        return _moments_via_quad(a, b, max_degree, _jacobi_weight_on_interval(a, b, m))

    return _moments_via_quad(a, b, max_degree, weight_func)

def moment_matched_quad_weights(
    nodes: np.ndarray,
    a: float,
    b: float,
    *,
    weight_func: Optional[Callable] = None,
    m: Optional[float] = None,
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
    weight_func : callable, optional
        Arbitrary weight function w(x).
    m : float, optional
        Exponent of the Jacobi weight (1-s(x)^2)^m, where s maps [a, b]
        onto [-1, 1].
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
    mu = compute_moments(a, b, degree, weight_func=weight_func, m=m)
    return np.linalg.lstsq(V.T, mu, rcond=None)[0]