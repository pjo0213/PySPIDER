import argparse
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.fft import dct
from scipy.integrate import quad


def chebyshev_lobatto_nodes(num_intervals: int) -> np.ndarray:
    """Return N+1 Chebyshev-Lobatto nodes on [-1, 1] for N=num_intervals."""
    if num_intervals < 1:
        raise ValueError("num_intervals must be >= 1")
    j = np.arange(num_intervals + 1, dtype=float)
    return np.cos(np.pi * j / num_intervals)


def chebyshev_coefficients_from_values(f_values: np.ndarray) -> np.ndarray:
    """Compute Chebyshev coefficients a_k from values at Lobatto nodes."""
    N = f_values.shape[0] - 1
    if N < 1:
        raise ValueError("At least two sample points (N>=1) are required")
    a = dct(f_values.astype(float), type=1) / N
    a[0] *= 0.5
    a[-1] *= 0.5
    return a


@lru_cache(maxsize=None)
def jacobi_weighted_moments_T(m: float, N: int, epsabs: float = 1e-13, epsrel: float = 1e-13) -> Tuple[np.ndarray, np.ndarray]:
    """Compute moments μ_k = ∫_{-1}^1 (1-x)^m (1+x)^m T_k(x) dx for k=0..N."""
    if m <= -1:
        raise ValueError("m must be > -1 for weighted quadrature with 'alg'.")
    if N < 0:
        raise ValueError("N must be >= 0")

    ks = np.arange(N + 1, dtype=int)
    mu = np.zeros(N + 1, dtype=float)
    for k in ks:
        if k % 2 == 1:
            mu[k] = 0.0
            continue

        def Tk(x: float, kk: int = k) -> float:
            return np.cos(kk * np.arccos(x))

        res, _ = quad(
            Tk,
            -1.0,
            1.0,
            weight="alg",
            wvar=(m, m),
            limit=200,
            epsabs=epsabs,
            epsrel=epsrel,
        )
        mu[k] = res

    return ks, mu


@lru_cache(maxsize=None)
def jacobi_weighted_moments_T_dct(m: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute moments μ_k via DCT on θ-grid: x = cos θ."""
    if m <= -1:
        raise ValueError("m must be > -1 for DCT-based moments.")
    if N < 0:
        raise ValueError("N must be >= 0")
    M = max(N, 1)
    js = np.arange(M + 1, dtype=float)
    theta = np.pi * js / M
    f_theta = np.sin(theta) ** (2.0 * m + 1.0)
    c = dct(f_theta, type=1) / M
    c[0] *= 0.5
    c[-1] *= 0.5
    mu = 0.5 * np.pi * c[: N + 1]
    mu[0] = np.pi * c[0]
    return np.arange(N + 1, dtype=int), mu


def integrate_weighted_clenshaw_curtis_from_values(
    x: np.ndarray,
    f_values: np.ndarray,
    m: float,
    epsabs: float = 1e-13,
    epsrel: float = 1e-13,
    moments: str = "dct",
) -> float:
    """Compute ∫_{-1}^1 (1-x^2)^m f(x) dx from Chebyshev-Lobatto samples."""
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
        _, mu = jacobi_weighted_moments_T(m, N, epsabs=epsabs, epsrel=epsrel)
    elif moments == "dct":
        _, mu = jacobi_weighted_moments_T_dct(m, N)
    else:
        raise ValueError("moments must be 'quadpack' or 'dct'")

    return float(np.dot(a, mu))


def demo_function(x: np.ndarray) -> np.ndarray:
    return np.exp(x) * (1.0 + 0.3 * x - 0.2 * x**2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ∫_{-1}^1 (1-x^2)^m f(x) dx on a Chebyshev grid using "
            "Clenshaw-Curtis with Jacobi moments."
        )
    )
    parser.add_argument("m", type=float, help="Positive exponent m in (1-x^2)^m")
    parser.add_argument("N", type=int, help="Number of Chebyshev intervals (N+1 points)")
    parser.add_argument("--demo", action="store_true", help="Use a built-in demo function f(x)")
    args = parser.parse_args()

    N = args.N
    m = args.m
    x = chebyshev_lobatto_nodes(N)
    if args.demo:
        fvals = demo_function(x)
    else:
        import sys

        data = sys.stdin.read().strip().split()
        if len(data) != N + 1:
            raise SystemExit(f"Expected {N+1} values from stdin, got {len(data)}")
        fvals = np.array([float(t) for t in data], dtype=float)

    integral_value = integrate_weighted_clenshaw_curtis_from_values(x, fvals, m)
    print(integral_value)


if __name__ == "__main__":
    main()
