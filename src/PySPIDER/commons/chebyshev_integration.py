from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.fft import dct
from scipy.integrate import quad

def chebyshev_lobatto_nodes(N: int) -> np.ndarray:
    """
    Return the Chebyshev-Lobatto nodes on [-1, 1].

    The N+1 nodes are x_j = cos(pi*j/N) for j=0..N, returned in descending
    order.

    Parameters
    ----------
    N : int
        Number of intervals; produces N+1 nodes. Must be >= 1.

    Returns
    -------
    np.ndarray
        Array of N+1 Chebyshev-Lobatto nodes.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    return np.cos(np.pi * np.arange(N + 1, dtype=float) / N)


def _validate_interval(a: float, b: float) -> None:
    if a > b:
        raise ValueError(f"interval endpoints must satisfy a <= b, got ({a}, {b})")

def mapped_chebyshev_nodes(N: int, a: float, b: float) -> np.ndarray:
    """
    Affinely map the Chebyshev-Lobatto nodes from [-1, 1] onto [a, b].

    Applies x -> (a+b)/2 + (b-a)/2 * x to the reference nodes and returns
    them sorted ascending.

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
    x = chebyshev_lobatto_nodes(N)
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * x)

def truncated_chebyshev_nodes(N: int, a: float, b: float) -> np.ndarray:
    """
    Return the Chebyshev-Lobatto nodes that fall within [a, b].

    The full reference grid is generated and only the subset lying in [a, b]
    (within a floating-point tolerance) is retained, sorted ascending.

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
    nodes = chebyshev_lobatto_nodes(N)
    tol = np.finfo(float).eps * max(abs(a), abs(b), 1.0)
    mask = (nodes >= a - tol) & (nodes <= b + tol)
    kept = np.sort(nodes[mask])
    if kept.size == 0:
        raise ValueError(
            f"No Chebyshev-Lobatto nodes from reference grid N={N} lie in [{a}, {b}]. "
            "Increase num_intervals or widen the interval."
        )
    return kept

def chebyshev_coefficients_from_values(f_values: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev coefficients from Lobatto-grid samples via a DCT-I.

    Given samples at the Chebyshev-Lobatto nodes, returns the coefficients
    a_k of the expansion f(x) = sum_{k=0}^{N} a_k T_k(x).

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

@lru_cache(maxsize=128)
def jacobi_weighted_moments_T_dct(m: float, max_degree: int) -> np.ndarray:
    """
    Compute Jacobi-weighted Chebyshev moments via a DCT.

    Returns the moments mu_k = integral_{-1}^{1} (1-x^2)^m T_k(x) dx. The
    substitution x = cos(theta) turns (1-x^2)^m dx into sin^{2m+1}(theta)
    dtheta, whose cosine-series coefficients (from a DCT-I) are the moments.

    For m = 0 the exact closed-form moments are returned. For -1 < m < 0 the
    theta-integrand is endpoint-singular or has too slowly decaying cosine
    coefficients for the DCT to be accurate, so the computation falls back to
    the QUADPACK backend.

    Parameters
    ----------
    m : float
        Weight exponent in (1-x^2)^m; must be > -1.
    max_degree : int
        Largest Chebyshev index k.

    Returns
    -------
    np.ndarray
        Read-only array of moments mu_0, ..., mu_max_degree.
    """
    if m <= -1:
        raise ValueError("m must be > -1 for DCT-based moments.")
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")
    if m == 0:
        mu = _unweighted_moments(-1.0, 1.0, max_degree)
        mu.setflags(write=False)
        return mu
    if m < 0:
        # For m < -0.5 sin^{2m+1}(theta) diverges at theta = 0, pi and DCT
        # sampling fails outright; for -0.5 <= m < 0 the cosine coefficients
        # decay slower than k^-2 and aliasing dominates. Use QUADPACK.
        return jacobi_weighted_moments_T(m, max_degree)
    # For integer m the cosine coefficients of sin^{2m+1}(theta) decay only
    # algebraically, ~k^-(2m+2); oversample the theta-grid so aliasing into
    # the first max_degree+1 coefficients is below roundoff for m >= 1 and
    # negligible for fractional m. The DCT cost is trivial and cached.
    M = max(2 * max_degree, 2048)
    theta = np.pi * np.arange(M + 1, dtype=float) / M
    f_theta = np.sin(theta) ** (2.0 * m + 1.0)
    c = chebyshev_coefficients_from_values(f_theta)
    mu = 0.5 * np.pi * c[: max_degree + 1]
    mu[0] = np.pi * c[0]
    # cached result; prevent callers from corrupting it
    mu.setflags(write=False)
    return mu

@lru_cache(maxsize=128)
def jacobi_weighted_moments_T(
    m: float, max_degree: int, epsabs: float = 1e-13, epsrel: float = 1e-13
) -> np.ndarray:
    """
    Compute Jacobi-weighted Chebyshev moments via QUADPACK.

    Returns the moments mu_k = integral_{-1}^{1} (1-x)^m (1+x)^m T_k(x) dx
    using scipy.integrate.quad with the 'alg' weight. Odd-order moments vanish
    by symmetry and are skipped.

    Parameters
    ----------
    m : float
        Weight exponent in (1-x^2)^m; must be > -1.
    max_degree : int
        Largest Chebyshev index k.
    epsabs : float, optional
        Absolute error tolerance for the quadrature.
    epsrel : float, optional
        Relative error tolerance for the quadrature.

    Returns
    -------
    np.ndarray
        Read-only array of moments mu_0, ..., mu_max_degree.
    """
    if m <= -1:
        raise ValueError("m must be > -1 for weighted quadrature with 'alg'.")
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")

    mu = np.zeros(max_degree + 1, dtype=float)
    for k in range(max_degree + 1):
        # (1-x^2)^m is even, so odd-order moments vanish.
        if k % 2 == 1:
            continue

        def Tk(x: float, kk: int = k) -> float:
            return np.cos(kk * np.arccos(x))

        res, _ = quad(
            Tk, -1.0, 1.0, weight="alg", wvar=(m, m), limit=200, epsabs=epsabs, epsrel=epsrel
        )
        mu[k] = res
    # cached result; prevent callers from corrupting it
    mu.setflags(write=False)
    return mu

def _unweighted_moments(a: float, b: float, max_degree: int) -> np.ndarray:
    """
    Compute the unweighted Chebyshev moments analytically.

    Returns mu_k = integral_a^b T_k(x) dx evaluated in closed form via the
    Chebyshev antiderivative recurrence.

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
    mu = np.zeros(max_degree + 1)
    mu[0] = b - a
    if max_degree >= 1:
        mu[1] = (b ** 2 - a ** 2) / 2.0
    for k in range(2, max_degree + 1):
        ckp1 = np.zeros(k + 2)
        ckp1[k + 1] = 1.0
        ckm1 = np.zeros(k)
        ckm1[k - 1] = 1.0
        anti_b = chebval(b, ckp1) / (2 * (k + 1)) - chebval(b, ckm1) / (2 * (k - 1))
        anti_a = chebval(a, ckp1) / (2 * (k + 1)) - chebval(a, ckm1) / (2 * (k - 1))
        mu[k] = anti_b - anti_a
    return mu

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
    envelope_m: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Chebyshev moments for a chosen weight on [a, b].

    Returns mu_k = integral_a^b w(x) T_k(x) dx. At most one weight
    specification may be given: weight_func for an arbitrary w(x), m for the
    Jacobi weight (1-x^2)^m, or envelope_m for ((x-a)(b-x))^envelope_m.
    With none, w(x) = 1.

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
        Exponent of the Jacobi weight (1-x^2)^m.
    envelope_m : float, optional
        Exponent of the endpoint envelope weight.

    Returns
    -------
    np.ndarray
        Array of moments mu_0, ..., mu_max_degree.
    """
    specs = sum(x is not None for x in [weight_func, m, envelope_m])
    if specs > 1:
        raise ValueError("Specify at most one of weight_func, m, envelope_m")

    if specs == 0:
        return _unweighted_moments(a, b, max_degree)

    if m is not None:
        if np.isclose(a, -1.0) and np.isclose(b, 1.0):
            return jacobi_weighted_moments_T_dct(m, max_degree)
        return _moments_via_quad(a, b, max_degree, lambda x: (1.0 - x ** 2) ** m)

    if envelope_m is not None:
        return _moments_via_quad(a, b, max_degree, lambda x: ((x - a) * (b - x)) ** envelope_m)

    return _moments_via_quad(a, b, max_degree, weight_func)

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

def _interval_cc_scale_and_ref_m(
    a: float,
    b: float,
    *,
    envelope_m: Optional[float] = None,
    jacobi_m: float = 0.0,
) -> tuple[float, float]:
    """Return (scale, reference_jacobi_m) for affine Clenshaw-Curtis on [a, b]."""
    if envelope_m is not None:
        em = float(envelope_m)
        return (0.5 * (b - a)) ** (2.0 * em + 1.0), em
    if np.isclose(a, -1.0) and np.isclose(b, 1.0):
        return 1.0, float(jacobi_m)
    return 0.5 * (b - a), 0.0

@lru_cache(maxsize=128)
def clenshaw_curtis_weights(
    num_intervals: int,
    m: float = 0.0,
    moments: str = "dct",
    epsabs: float = 1e-13,
    epsrel: float = 1e-13,
) -> np.ndarray:
    """
    Compute Clenshaw-Curtis quadrature weights on the full Lobatto grid.

    Returns weights w_j such that sum_j w_j f(x_j) = integral_{-1}^{1}
    (1-x^2)^m f(x) dx for f sampled at the Lobatto nodes x_j = cos(pi*j/N),
    j = 0..N (descending order). The weights are the DCT-I coefficient map
    contracted with the weighted moments, so a single dot product reproduces
    integrate_weighted_clenshaw_curtis_from_values exactly. Because the
    Jacobi weight is even, the weight vector is symmetric and applies
    unchanged to samples stored in ascending node order.

    Parameters
    ----------
    num_intervals : int
        Number of grid intervals N; the grid has N+1 nodes.
    m : float, optional
        Weight exponent in (1-x^2)^m.
    moments : str, optional
        Moment backend, 'dct' or 'quadpack'.
    epsabs : float, optional
        Absolute tolerance for QUADPACK moments.
    epsrel : float, optional
        Relative tolerance for QUADPACK moments.

    Returns
    -------
    np.ndarray
        Read-only array of N+1 quadrature weights.
    """
    N = num_intervals
    if N < 1:
        raise ValueError("num_intervals must be >= 1")
    if moments == "quadpack":
        mu = jacobi_weighted_moments_T(m, N, epsabs=epsabs, epsrel=epsrel)
    elif moments == "dct":
        mu = jacobi_weighted_moments_T_dct(m, N)
    else:
        raise ValueError("moments must be 'quadpack' or 'dct'")

    # Coefficient map C[k, j]: a_k = sum_j C[k, j] f_j (DCT-I normalization
    # used by chebyshev_coefficients_from_values); weights are C^T mu.
    j = np.arange(N + 1)
    C = np.cos(np.pi * np.outer(j, j) / N) / N
    C[:, 1:-1] *= 2.0
    C[0, :] *= 0.5
    C[-1, :] *= 0.5
    w = C.T @ mu
    w.setflags(write=False)
    return w

@lru_cache(maxsize=128)
def clenshaw_curtis_weights_on_interval(
    a: float,
    b: float,
    num_intervals: int,
    *,
    envelope_m: Optional[float] = None,
    m: float = 0.0,
    moments: str = "dct",
    epsabs: float = 1e-13,
    epsrel: float = 1e-13,
) -> np.ndarray:
    """
    Clenshaw-Curtis weights on the full Lobatto grid mapped to [a, b].

    Returns weights w_j such that sum_j w_j f(x_j) approximates

        integral_a^b f(x) dx                           (default),
        integral_a^b ((x-a)(b-x))^m f(x) dx            (envelope_m=m), or
        integral_{-1}^{1} (1-x^2)^m f(x) dx            (a=-1, b=1, m=m).

    Uses the O(N log N) reference Clenshaw-Curtis weights with the affine
    Jacobian factor; CC weights are symmetric so ascending or descending sample
    order is equivalent.
    """
    _validate_interval(a, b)
    scale, ref_m = _interval_cc_scale_and_ref_m(a, b, envelope_m=envelope_m, jacobi_m=m)
    w = scale * clenshaw_curtis_weights(
        num_intervals, m=ref_m, moments=moments, epsabs=epsabs, epsrel=epsrel
    )
    return w

def _chebyshev_basis_matrix(x: np.ndarray, max_degree: int) -> np.ndarray:
    """
    Evaluate Chebyshev polynomials at nodes via the recurrence.

    Builds the matrix whose columns are T_0, T_1, ..., T_max_degree evaluated
    at the given nodes.

    Parameters
    ----------
    x : np.ndarray
        Nodes at which to evaluate the basis.
    max_degree : int
        Largest Chebyshev index to include.

    Returns
    -------
    np.ndarray
        Matrix of shape (len(x), max_degree + 1).
    """
    M = len(x)
    D = max_degree + 1
    V = np.zeros((M, D))
    V[:, 0] = 1.0
    if D > 1:
        V[:, 1] = x
    for k in range(2, D):
        V[:, k] = 2.0 * x * V[:, k - 1] - V[:, k - 2]
    return V

def quadrature_weights(
    nodes: np.ndarray,
    a: float,
    b: float,
    *,
    weight_func: Optional[Callable] = None,
    m: Optional[float] = None,
    envelope_m: Optional[float] = None,
    degree: Optional[int] = None,
) -> np.ndarray:
    """
    Compute quadrature weights for arbitrary nodes by moment matching.

    Solves the (least-squares) moment-matching system V^T w = mu, where V is
    the Chebyshev basis matrix at nodes and mu are the weighted moments on
    [a, b].

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
        Exponent of the Jacobi weight (1-x^2)^m.
    envelope_m : float, optional
        Exponent of the endpoint envelope weight.
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
    if M == 0:
        raise ValueError("quadrature_weights requires at least one node.")
    if M == 1:
        raise ValueError("quadrature_weights requires at least two nodes (degree >= 1).")

    if degree is None:
        degree = M - 1

    V = _chebyshev_basis_matrix(nodes, degree)
    mu = compute_moments(a, b, degree, weight_func=weight_func, m=m, envelope_m=envelope_m)
    return np.linalg.lstsq(V.T, mu, rcond=None)[0]