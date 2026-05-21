from typing import Optional, Callable
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.integrate import quad
from scipy.fft import dct


def chebyshev_lobatto_nodes(N: int) -> np.ndarray:
    """Return N+1 Chebyshev-Lobatto nodes on [-1, 1] in descending order."""
    if N < 1:
        raise ValueError("N must be >= 1")
    return np.cos(np.pi * np.arange(N + 1, dtype=float) / N)


def mapped_chebyshev_nodes(N: int, a: float, b: float) -> np.ndarray:
    """Return N+1 Chebyshev-Lobatto nodes affinely mapped to [a, b]."""
    x = chebyshev_lobatto_nodes(N)
    return np.sort(0.5 * (a + b) + 0.5 * (b - a) * x)


def truncated_chebyshev_nodes(N: int, a: float, b: float) -> np.ndarray:
    """Return Chebyshev-Lobatto nodes that lie in [a, b], sorted ascending."""
    nodes = chebyshev_lobatto_nodes(N)
    tol = np.finfo(float).eps * max(abs(a), abs(b), 1.0)
    mask = (nodes >= a - tol) & (nodes <= b + tol)
    return np.sort(nodes[mask])


def _chebyshev_basis_matrix(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Evaluate T_0, ..., T_{max_degree} at nodes x via recurrence."""
    M = len(x)
    D = max_degree + 1
    V = np.zeros((M, D))
    V[:, 0] = 1.0
    if D > 1:
        V[:, 1] = x
    for k in range(2, D):
        V[:, k] = 2.0 * x * V[:, k - 1] - V[:, k - 2]
    return V


def _unweighted_moments(a: float, b: float, max_degree: int) -> np.ndarray:
    """Analytic moments mu_k = ∫_a^b T_k(x) dx."""
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


def _jacobi_ref_moments_dct(m: float, max_degree: int) -> np.ndarray:
    """Moments mu_k = ∫_{-1}^{1} (1 - x^2)^m T_k(x) dx via DCT."""
    M = max(max_degree, int(4 * abs(m)) + 4, 1)
    theta = np.pi * np.arange(M + 1, dtype=float) / M
    f_theta = np.sin(theta) ** (2.0 * m + 1.0)
    c = dct(f_theta, type=1) / M
    c[0] *= 0.5
    c[-1] *= 0.5
    mu = 0.5 * np.pi * c[:max_degree + 1]
    mu[0] = np.pi * c[0]
    return mu


def _moments_via_quad(a: float, b: float, max_degree: int, w_fn: Callable[[float], float]) -> np.ndarray:
    """Moments mu_k = ∫_a^b w(x) T_k(x) dx via scipy.integrate.quad."""
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
    """Compute moments mu_k = ∫_a^b w(x) T_k(x) dx for k = 0, ..., max_degree."""
    specs = sum(x is not None for x in [weight_func, m, envelope_m])
    if specs > 1:
        raise ValueError("Specify at most one of weight_func, m, envelope_m")

    if specs == 0:
        return _unweighted_moments(a, b, max_degree)

    if m is not None:
        if np.isclose(a, -1.0) and np.isclose(b, 1.0):
            return _jacobi_ref_moments_dct(m, max_degree)
        return _moments_via_quad(a, b, max_degree, lambda x: (1.0 - x ** 2) ** m)

    if envelope_m is not None:
        return _moments_via_quad(a, b, max_degree, lambda x: ((x - a) * (b - x)) ** envelope_m)

    return _moments_via_quad(a, b, max_degree, weight_func)


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
    """Compute quadrature weights for the given nodes in [a, b]."""
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
    """Compute ∫_a^b w(x) f(x) dx from nodal values."""
    nodes = np.asarray(nodes, dtype=float)
    f_values = np.asarray(f_values, dtype=float)

    if a is None:
        a = float(nodes.min())
    if b is None:
        b = float(nodes.max())

    w = quadrature_weights(nodes, a, b, weight_func=weight_func, m=m, envelope_m=envelope_m, degree=degree)
    return float(np.dot(w, f_values))
