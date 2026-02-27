import argparse
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.fft import dct
from scipy.integrate import quad


def chebyshev_lobatto_nodes(num_intervals: int) -> np.ndarray:
	"""Return N+1 Chebyshev–Lobatto nodes on [-1, 1] for N=num_intervals.

	Parameters
	----------
	num_intervals: int
		N in x_j = cos(pi * j / N), j=0..N. Must be >= 1.

	Returns
	-------
	np.ndarray
		Array of shape (N+1,) with nodes in decreasing order from 1 to -1.
	"""
	if num_intervals < 1:
		raise ValueError("num_intervals must be >= 1")
	j = np.arange(num_intervals + 1, dtype=float)
	return np.cos(np.pi * j / num_intervals)


def chebyshev_coefficients_from_values(f_values: np.ndarray) -> np.ndarray:
	"""Compute Chebyshev coefficients a_k from values at Chebyshev–Lobatto nodes.

	Given samples f(cos(j*pi/N)) for j=0..N, compute coefficients a_k such that

		f(x) ≈ sum_{k=0}^{N} a_k T_k(x).

	This uses a DCT-I on raw values with normalization a = dct(f, type=1)/N and
	then halves a_0 and a_N, which yields the standard Chebyshev series
	coefficients for direct summation without endpoint 1/2 factors.

	Parameters
	----------
	f_values: np.ndarray
		Array of length N+1 with samples at Chebyshev–Lobatto nodes.

	Returns
	-------
	np.ndarray
		Chebyshev coefficients a_k for k=0..N for f(x) = sum a_k T_k(x).
	"""
	N = f_values.shape[0] - 1
	if N < 1:
		raise ValueError("At least two sample points (N>=1) are required")
	# DCT-I normalization to obtain series without endpoint factors
	a = dct(f_values.astype(float), type=1) / N
	a[0] *= 0.5
	a[-1] *= 0.5
	return a


@lru_cache(maxsize=None)
def jacobi_weighted_moments_T(m: float, N: int, epsabs: float = 1e-13, epsrel: float = 1e-13) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute moments μ_k = ∫_{-1}^1 (1-x)^m (1+x)^m T_k(x) dx for k=0..N.

	Moments are computed using SciPy's quad with weight='alg' (QUADPACK dqaws),
	which exactly mirrors SciPy's behavior for algebraic endpoint weights.

	Parameters
	----------
	m: float
		Positive exponent for the Jacobi weight. Requires m > -1 to satisfy dqaws.
	N: int
		Maximum polynomial index.
	epsabs: float
		Absolute tolerance for quad.
	epsrel: float
		Relative tolerance for quad.

	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(k_values, mu_values) where k_values = [0,1,...,N] and mu_values has the same length.
	"""
	if m <= -1:
		raise ValueError("m must be > -1 for weighted quadrature with 'alg'.")
	if N < 0:
		raise ValueError("N must be >= 0")

	ks = np.arange(N + 1, dtype=int)
	mu = np.zeros(N + 1, dtype=float)

	# (1-x)^m (1+x)^m is even; T_k is odd for odd k, hence μ_k = 0 for odd k.
	for k in ks:
		if k % 2 == 1:
			mu[k] = 0.0
			continue
		def Tk(x: float, kk: int = k) -> float:
			# Chebyshev polynomial T_k(x) = cos(k arccos x)
			return np.cos(kk * np.arccos(x))
		res, _ = quad(
			Tk,
			-1.0,
			1.0,
			weight='alg',
			wvar=(m, m),
			limit=200,
			epsabs=epsabs,
			epsrel=epsrel,
		)
		mu[k] = res

	return ks, mu


@lru_cache(maxsize=None)
def jacobi_weighted_moments_T_dct(m: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute moments μ_k via DCT on θ-grid: x = cos θ, f(θ) = sin^{2m+1} θ.

	Cosine-series coefficients c_k of f(θ) are computed on θ_j = jπ/N using
	DCT-I with normalization c = dct(f, type=1)/N and endpoint 1/2 adjustments.
	Then μ_0 = π c_0 and μ_k = (π/2) c_k for k > 0.
	"""
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
	# First N+1 coefficients provide moments; k=0 uses π c0, k>0 uses (π/2) c_k
	mu = 0.5 * np.pi * c[: N + 1]
	mu[0] = np.pi * c[0]
	return np.arange(N + 1, dtype=int), mu


def integrate_weighted_clenshaw_curtis_from_values(x: np.ndarray, f_values: np.ndarray, m: float,
														epsabs: float = 1e-13, epsrel: float = 1e-13,
														moments: str = 'dct') -> float:
	"""Compute ∫_{-1}^1 (1-x^2)^m f(x) dx from Chebyshev–Lobatto samples using CC.

	The algorithm:
	1) Compute Chebyshev T_k coefficients a_k from samples via a DCT-I.
	2) Compute weighted moments μ_k = ∫ (1-x)^m(1+x)^m T_k(x) dx using quad(weight='alg').
	3) Combine as I ≈ a_0/2 μ_0 + sum_{k=1}^{N-1} a_k μ_k + a_N/2 μ_N.

	Parameters
	----------
	x: np.ndarray
		Chebyshev–Lobatto nodes of length N+1 on [-1, 1].
	f_values: np.ndarray
		Function values at x.
	m: float
		Positive exponent for (1-x^2)^m.
	epsabs, epsrel: float
		Quadrature tolerances for the moment computations.

	Returns
	-------
	float
		Approximation to the desired weighted integral.
	"""
	if x.shape != f_values.shape:
		raise ValueError("x and f_values must have the same shape")
	N = x.shape[0] - 1
	# Validate that x are Chebyshev–Lobatto nodes (within a tolerance).
	if N < 1:
		raise ValueError("Require at least N>=1 (two points)")
	expected = chebyshev_lobatto_nodes(N)
	if not np.allclose(x, expected, rtol=0, atol=1e-12):
		raise ValueError("x must be Chebyshev–Lobatto nodes cos(pi*j/N), j=0..N")

	# Chebyshev coefficients a_k
	a = chebyshev_coefficients_from_values(f_values)
	# Weighted moments μ_k
	if moments == 'quadpack':
		_, mu = jacobi_weighted_moments_T(m, N, epsabs=epsabs, epsrel=epsrel)
	elif moments == 'dct':
		_, mu = jacobi_weighted_moments_T_dct(m, N)
	else:
		raise ValueError("moments must be 'quadpack' or 'dct'")

	# Combine directly: ∫ w(x) f(x) dx = Σ a_k μ_k
	return float(np.dot(a, mu))


def integrate_weighted_clenshaw_curtis_interval_from_values(a: float, b: float,
																	 t_nodes: np.ndarray,
																	 f_at_x: np.ndarray,
																	 m: float,
																	 moments: str = 'dct') -> float:
	"""Compute ∫_{a}^{b} (a-x)^m (x-b)^m f(x) dx using CC on Chebyshev t-nodes.

	We map x = (a+b)/2 + (b-a)/2 t with t ∈ [-1,1], so
	(a-x)^m (x-b)^m dx = ((b-a)/2)^{2m+1} (1-t^2)^m dt.
	The integral becomes scaling * ∫_{-1}^1 (1-t^2)^m f(x(t)) dt, which we
	compute by Chebyshev expansion in t and precomputed moments μ_k for (1-t^2)^m.
	"""
	if t_nodes.shape != f_at_x.shape:
		raise ValueError("t_nodes and f_at_x must have the same shape")
	N = t_nodes.shape[0] - 1
	if N < 1:
		raise ValueError("Require at least N>=1 (two points)")
	# Validate Chebyshev–Lobatto nodes in t
	expected = chebyshev_lobatto_nodes(N)
	if not np.allclose(t_nodes, expected, rtol=0, atol=1e-12):
		raise ValueError("t_nodes must be cos(pi*j/N), j=0..N")

	# Chebyshev coefficients in t
	a_coeff = chebyshev_coefficients_from_values(f_at_x)
	# Moments for (1-t^2)^m
	if moments == 'dct':
		_, mu = jacobi_weighted_moments_T_dct(m, N)
	elif moments == 'quadpack':
		_, mu = jacobi_weighted_moments_T(m, N)
	else:
		raise ValueError("moments must be 'dct' or 'quadpack'")
	acc_t = float(np.dot(a_coeff, mu))
	# Scaling factor
	scale = ((b - a) * 0.5) ** (2.0 * m + 1.0)
	return scale * acc_t


def jacobi_weighted_moments_T_on_interval(a: float, b: float, m: float, N: int,
														 epsabs: float = 1e-13, epsrel: float = 1e-13) -> tuple[np.ndarray, np.ndarray]:
	"""Compute μ_k = ∫_{a}^{b} (a-x)^m (x-b)^m T_k(x) dx for k=0..N via QUADPACK.

	Uses scipy.integrate.quad with weight='alg' over [a,b], wvar=(m,m).
	"""
	if N < 0:
		raise ValueError("N must be >= 0")
	ks = np.arange(N + 1, dtype=int)
	mu = np.zeros(N + 1, dtype=float)
	for k in ks:
		def Tk(x: float, kk: int = k) -> float:
			return np.cos(kk * np.arccos(x))
		res, _ = quad(
			Tk,
			a,
			b,
			weight='alg',
			wvar=(m, m),
			limit=200,
			epsabs=epsabs,
			epsrel=epsrel,
		)
		mu[k] = res
	return ks, mu


def integrate_weighted_clenshaw_curtis_piecewise_from_values(x_nodes: np.ndarray,
															 f_values: np.ndarray,
															 a: float,
															 b: float,
															 m: float,
															 epsabs: float = 1e-13,
															 epsrel: float = 1e-13) -> float:
	"""Compute ∫_{-1}^{1} w(x) f(x) dx with w(x)=(a-x)^m(x-b)^m on [a,b], 0 else.

	- x_nodes must be Chebyshev–Lobatto nodes on [-1,1]
	- f_values are samples of f at x_nodes
	- We compute Chebyshev coefficients of f and moments μ_k over [a,b]
	"""
	if x_nodes.shape != f_values.shape:
		raise ValueError("x_nodes and f_values must have the same shape")
	N = x_nodes.shape[0] - 1
	if N < 1:
		raise ValueError("Require at least N>=1 (two points)")
	# Validate Chebyshev–Lobatto nodes on [-1,1]
	expected = chebyshev_lobatto_nodes(N)
	if not np.allclose(x_nodes, expected, rtol=0, atol=1e-12):
		raise ValueError("x_nodes must be cos(pi*j/N), j=0..N")
	# Coefficients of f(x) in T_k basis
	a_coeff = chebyshev_coefficients_from_values(f_values)
	# Moments over [a,b]
	_, mu = jacobi_weighted_moments_T_on_interval(a, b, m, N, epsabs=epsabs, epsrel=epsrel)
	return float(np.dot(a_coeff, mu))


@lru_cache(maxsize=None)
def plain_moments_T(N: int, epsabs: float = 1e-13, epsrel: float = 1e-13) -> tuple[np.ndarray, np.ndarray]:
	"""Compute ν_k = ∫_{-1}^1 T_k(x) dx for k=0..N using quad (no special weight)."""
	if N < 0:
		raise ValueError("N must be >= 0")
	ks = np.arange(N + 1, dtype=int)
	nu = np.zeros(N + 1, dtype=float)
	for k in ks:
		def Tk(x: float, kk: int = k) -> float:
			return np.cos(kk * np.arccos(x))
		res, _ = quad(Tk, -1.0, 1.0, limit=200, epsabs=epsabs, epsrel=epsrel)
		nu[k] = res
	return ks, nu


def integrate_indicator_weighted_from_values(x_nodes: np.ndarray,
											 f_values: np.ndarray,
											 a: float,
											 b: float,
											 m: float) -> float:
	"""Integrate g(x)=f(x)*(a-x)^m(x-b)^m on [a,b], zero elsewhere, via CC on [-1,1].

	- Uses only the provided values f(x_nodes) on the fixed Chebyshev grid
	- Forms g_values by multiplying by the piecewise weight (zero outside [a,b])
	- Integrates g over [-1,1] using Chebyshev coefficients and plain moments ν_k
	"""
	if x_nodes.shape != f_values.shape:
		raise ValueError("x_nodes and f_values must have the same shape")
	N = x_nodes.shape[0] - 1
	if N < 1:
		raise ValueError("Require at least N>=1 (two points)")
	# Validate Chebyshev nodes
	expected = chebyshev_lobatto_nodes(N)
	if not np.allclose(x_nodes, expected, rtol=0, atol=1e-12):
		raise ValueError("x_nodes must be cos(pi*j/N), j=0..N")
	# Build piecewise weight and g values
	mask = (x_nodes >= a) & (x_nodes <= b)
	w = np.zeros_like(x_nodes)
	w[mask] = (a - x_nodes[mask]) ** m * (x_nodes[mask] - b) ** m
	g_values = f_values * w
	# Chebyshev coefficients of g and plain moments
	a_g = chebyshev_coefficients_from_values(g_values)
	_, nu = plain_moments_T(N)
	return float(np.dot(a_g, nu))


"""
NOTE: All interval-weighted integrators require f to be supplied as values at
Chebyshev–Lobatto nodes. Do not pass callables; do not resample f.
"""


def demo_function(x: np.ndarray) -> np.ndarray:
	"""A smooth test function on [-1,1] for demonstration purposes."""
	return np.exp(x) * (1.0 + 0.3 * x - 0.2 * x**2)


def main() -> None:
	parser = argparse.ArgumentParser(description=(
		"Compute ∫_{-1}^1 (1-x^2)^m f(x) dx on a Chebyshev grid using Clenshaw–Curtis, "
		"with Jacobi moments evaluated by SciPy/QUADPACK (dqaws)"
	))
	parser.add_argument("m", type=float, help="Positive exponent m in (1-x^2)^m")
	parser.add_argument("N", type=int, help="Number of Chebyshev intervals (grid has N+1 points)")
	parser.add_argument("--demo", action="store_true", help="Use a built-in demo function f(x)")
	args = parser.parse_args()

	N = args.N
	m = args.m
	x = chebyshev_lobatto_nodes(N)
	if args.demo:
		fvals = demo_function(x)
	else:
		# If not in demo mode, read f(x) values from stdin as whitespace-separated numbers
		# corresponding to x in decreasing order (cos(pi*j/N), j=0..N).
		import sys
		data = sys.stdin.read().strip().split()
		if len(data) != N + 1:
			raise SystemExit(f"Expected {N+1} values from stdin, got {len(data)}")
		fvals = np.array([float(t) for t in data], dtype=float)

	integral_value = integrate_weighted_clenshaw_curtis_from_values(x, fvals, m)
	print(integral_value)