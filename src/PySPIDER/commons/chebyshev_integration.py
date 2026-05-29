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

def mapped_chebyshev_nodes(N: int, a: float, b: float) -> np.ndarray:
    """
    Affinely map the Chebyshev-Lobatto nodes onto [a, b].

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
    x = chebyshev_lobatto_nodes(N)
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * x)

#if useful
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
    nodes = chebyshev_lobatto_nodes(N)
    tol = np.finfo(float).eps * max(abs(a), abs(b), 1.0)
    mask = (nodes >= a - tol) & (nodes <= b + tol)
    return np.sort(nodes[mask])

def chebyshev_coefficients_from_values(f_values: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev coefficients from Lobatto-grid samples via a DCT-I.

    Given samples at the Chebyshev-Lobatto nodes, returns the coefficients
    a_k of the expansion f(x) = sum_{k=0}^{N} a_k T_k(x). The endpoint
    coefficients are halved so the series can be summed directly without
    explicit endpoint factors.

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
    M = max(max_degree, int(4 * abs(m)) + 4, 1)
    theta = np.pi * np.arange(M + 1, dtype=float) / M
    f_theta = np.sin(theta) ** (2.0 * m + 1.0)
    c = dct(f_theta, type=1) / M
    c[0] *= 0.5
    c[-1] *= 0.5
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

def _full_lobatto_reference_degree(nodes: np.ndarray, a: float, b: float) -> Optional[int]:
    """
    Return N when nodes are Chebyshev-Lobatto on [-1, 1], else None.

    Parameters
    ----------
    nodes : np.ndarray
        Candidate quadrature nodes.
    a : float
        Left integration endpoint.
    b : float
        Right integration endpoint.

    Returns
    -------
    int or None
        Reference interval count N with N+1 nodes, or None if not a match.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size < 2:
        return None
    if not (np.isclose(a, -1.0) and np.isclose(b, 1.0)):
        return None
    N = nodes.shape[0] - 1
    expected = chebyshev_lobatto_nodes(N)
    if np.allclose(nodes, expected, rtol=0, atol=1e-12):
        return N
    return None

def integrate_weighted_clenshaw_curtis_from_values(
    x: np.ndarray,
    f_values: np.ndarray,
    m: float,
    epsabs: float = 1e-13,
    epsrel: float = 1e-13,
    moments: str = "dct",
) -> float:
    """
    Integrate a Jacobi-weighted function from full Lobatto-grid samples.

    Computes integral_{-1}^{1} (1-x^2)^m f(x) dx as the dot product of the
    Chebyshev coefficients of f with the weighted moments. The samples must lie
    on a complete Lobatto grid cos(pi*j/N).

    Parameters
    ----------
    x : np.ndarray
        Chebyshev-Lobatto nodes of length N+1.
    f_values : np.ndarray
        Samples of f at x.
    m : float
        Weight exponent in (1-x^2)^m.
    epsabs : float, optional
        Absolute tolerance for QUADPACK moments.
    epsrel : float, optional
        Relative tolerance for QUADPACK moments.
    moments : str, optional
        Moment backend, 'dct' or 'quadpack'.

    Returns
    -------
    float
        Approximation to the weighted integral.
    """
    x = np.asarray(x, dtype=float)
    f_values = np.asarray(f_values, dtype=float)
    if x.shape != f_values.shape:
        raise ValueError("x and f_values must have the same shape")
    N = x.shape[0] - 1
    if N < 1:
        raise ValueError("Require at least N>=1 (two points)")
    expected = chebyshev_lobatto_nodes(N)
    if not np.allclose(x, expected, rtol=0, atol=1e-12):
        raise ValueError("x must be Chebyshev-Lobatto nodes cos(pi*j/N), j=0..N")

    a = chebyshev_coefficients_from_values(f_values)
    if moments == "quadpack":
        mu = jacobi_weighted_moments_T(m, N, epsabs=epsabs, epsrel=epsrel)
    elif moments == "dct":
        mu = jacobi_weighted_moments_T_dct(m, N)
    else:
        raise ValueError("moments must be 'quadpack' or 'dct'")

    return float(np.dot(a, mu))

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
    M = len(nodes)
    if M == 0:
        return np.array([], dtype=float)

    if degree is None:
        degree = M - 1

    V = _chebyshev_basis_matrix(nodes, degree)
    mu = compute_moments(a, b, degree, weight_func=weight_func, m=m, envelope_m=envelope_m)
    return np.linalg.lstsq(V.T, mu, rcond=None)[0]

def integrate(
    nodes: np.ndarray,
    f_values: np.ndarray,
    a: Optional[float] = None,
    b: Optional[float] = None,
    *,
    weight_func: Optional[Callable] = None,
    m: Optional[float] = None,
    envelope_m: Optional[float] = None,
    degree: Optional[int] = None,
) -> float:
    """
    Integrate a function from nodal values on arbitrary nodes in [a, b].

    Computes integral_a^b w(x) f(x) dx by forming the quadrature weights for
    nodes and contracting them with the sample values.

    Parameters
    ----------
    nodes : np.ndarray
        Quadrature nodes in [a, b].
    f_values : np.ndarray
        Samples of f at nodes.
    a : float, optional
        Left endpoint; defaults to min(nodes).
    b : float, optional
        Right endpoint; defaults to max(nodes).
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
    float
        Approximation to the weighted integral.
    """
    nodes = np.asarray(nodes, dtype=float)
    f_values = np.asarray(f_values, dtype=float)

    if a is None:
        a = float(nodes.min())
    if b is None:
        b = float(nodes.max())

    # Full Lobatto grid on [-1, 1]: DCT Clenshaw-Curtis (Jacobi weight only).
    if weight_func is None and envelope_m is None:
        if _full_lobatto_reference_degree(nodes, a, b) is not None:
            jacobi_m = 0.0 if m is None else float(m)
            return integrate_weighted_clenshaw_curtis_from_values(nodes, f_values, jacobi_m)

    w = quadrature_weights(nodes, a, b, weight_func=weight_func, m=m, envelope_m=envelope_m, degree=degree)
    return float(np.dot(w, f_values))
