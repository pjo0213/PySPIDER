"""PySPIDER's array-integration dispatcher: `int_arr` and its per-scheme, per-axis helpers.

This module is the PySPIDER-specific glue layer: given a numpy array (a
term-times-weight product evaluated on an `IntegrationDomain`), an
`integration_scheme` string, and a `scheme_options` dict, it decides - axis
by axis - which quadrature nodes/weights to use and contracts them against
the data. All of the actual numerical quadrature (node generation, weight
computation) lives in `quadrature_schemes.py`; this module only owns option
parsing, per-axis dispatch, and the `np.tensordot` contractions.

`AbstractDataset.eval_on_domain` (in `process_library_terms.py`) is the only
caller: it calls `int_arr(term_weight_product, dxs=self.dxs,
scheme=self.integration_scheme, scheme_options=self.integration_options)`.
See the `integration_scheme`/`integration_options` field comments on
`AbstractDataset` for the full list of supported scheme strings and options.
"""

import numpy as np

from .quadrature_schemes import (
    clenshaw_curtis_weights_on_interval,
    gauss_legendre_nodes_and_weights,
    mapped_chebyshev_nodes,
    newton_cotes_weights,
    quadrature_weights,
    simpson_integrate,
    spline_quadrature_weights,
    truncated_chebyshev_nodes,
    _mapped_lobatto_reference_degree,
)


def _get_axis_option(value, axis, ndim):
    # per-axis options may be given as a list/tuple with one entry per axis
    if isinstance(value, (list, tuple)) and len(value) == ndim:
        return value[axis]
    return value


def _get_interval(interval_spec, axis, ndim):
    # interval_spec is either a single (a, b) pair or a list of per-axis pairs
    if interval_spec is None:
        return -1.0, 1.0
    if len(interval_spec) == 2 and all(np.isscalar(x) for x in interval_spec):
        a, b = interval_spec
    else:
        a, b = interval_spec[axis]
    return float(a), float(b)


def _chebyshev_axis_nodes(axis, ndim, axis_size, scheme_options):
    """Nodes for Chebyshev quadrature along one logical axis."""
    interval_spec = scheme_options.get("intervals", scheme_options.get("interval"))
    a, b = _get_interval(interval_spec, axis, ndim)
    nodes_spec = _get_axis_option(scheme_options.get("nodes"), axis, ndim)
    if nodes_spec is not None:
        return np.asarray(nodes_spec, dtype=float), a, b

    n_intervals_spec = _get_axis_option(scheme_options.get("num_intervals"), axis, ndim)
    if n_intervals_spec is None:
        if not (np.isclose(a, -1.0) and np.isclose(b, 1.0)):
            raise ValueError(
                f"Axis {axis}: integrating over the truncated interval [{a}, {b}] "
                "requires either explicit 'nodes' or 'num_intervals' (the interval "
                "count of the underlying reference Lobatto grid) in "
                "integration_options; the node positions cannot be inferred from "
                "the truncated data length alone."
            )
        n_intervals = axis_size - 1
    else:
        n_intervals = int(n_intervals_spec)
    return truncated_chebyshev_nodes(n_intervals, a, b), a, b


def _integrate_axis_truncated(arr, axis, ndim, scheme_options, original_shape):
    truncated_axes = scheme_options.get("truncated_axes", [0])
    if axis not in truncated_axes:
        return np.trapezoid(arr, axis=0)

    axis_size = original_shape[axis]
    nodes, a, b = _chebyshev_axis_nodes(axis, ndim, axis_size, scheme_options)
    if nodes.shape[0] != arr.shape[0]:
        raise ValueError(
            f"Along axis {axis}, array length {arr.shape[0]} does not match "
            f"{nodes.shape[0]} Chebyshev nodes. Pass explicit 'nodes' in "
            "integration_options or slice data to the truncated node set."
        )

    weight_func = _get_axis_option(scheme_options.get("weight_func"), axis, ndim)
    m = _get_axis_option(scheme_options.get("m"), axis, ndim)
    envelope_m = _get_axis_option(scheme_options.get("envelope_m"), axis, ndim)
    degree = _get_axis_option(scheme_options.get("degree"), axis, ndim)

    # The quadrature is linear in the samples, so compute the weight vector
    # once per axis and contract, instead of re-solving the moment-matching
    # system for every 1-D slice.
    w = quadrature_weights(
        nodes, a, b, weight_func=weight_func, m=m, envelope_m=envelope_m, degree=degree
    )
    return np.tensordot(w, arr, axes=(0, 0))


def _integrate_axis_clenshaw_curtis(arr, axis, ndim, scheme_options, original_shape):
    """Integrate along one axis with DCT Clenshaw-Curtis on a full mapped Lobatto grid."""
    lobatto_axes = scheme_options.get("lobatto_axes", [0])
    if axis not in lobatto_axes:
        return np.trapezoid(arr, axis=0)

    interval_spec = scheme_options.get("intervals", scheme_options.get("interval"))
    a, b = _get_interval(interval_spec, axis, ndim)

    nodes_spec = _get_axis_option(scheme_options.get("nodes"), axis, ndim)
    if nodes_spec is None:
        n_intervals_spec = _get_axis_option(scheme_options.get("num_intervals"), axis, ndim)
        n_intervals = (
            original_shape[axis] - 1 if n_intervals_spec is None else int(n_intervals_spec)
        )
        nodes = mapped_chebyshev_nodes(n_intervals, a, b)
    else:
        nodes = np.asarray(nodes_spec, dtype=float)
        if _mapped_lobatto_reference_degree(nodes, a, b) is None:
            raise ValueError(
                f"Axis {axis}: 'clenshaw-curtis' requires the full mapped "
                f"Chebyshev-Lobatto node set on [{a}, {b}] (ascending or "
                "descending). Use 'truncated-cc-grid' for partial node sets."
            )

    if nodes.shape[0] != arr.shape[0]:
        raise ValueError(
            f"Along axis {axis}, array length {arr.shape[0]} does not match "
            f"{nodes.shape[0]} Lobatto nodes."
        )

    jacobi_m = _get_axis_option(scheme_options.get("m"), axis, ndim)
    if jacobi_m is None:
        jacobi_m = 0.0
    envelope_m = _get_axis_option(scheme_options.get("envelope_m"), axis, ndim)
    moments = scheme_options.get("moments", "dct")
    epsabs = scheme_options.get("epsabs", 1e-13)
    epsrel = scheme_options.get("epsrel", 1e-13)

    w = clenshaw_curtis_weights_on_interval(
        a,
        b,
        nodes.shape[0] - 1,
        envelope_m=envelope_m,
        m=float(jacobi_m),
        moments=moments,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return np.tensordot(w, arr, axes=(0, 0))


def _integrate_axis_simpson(arr, axis, ndim, scheme_options, original_shape):
    """Composite Simpson's rule on a unit-spacing grid (uniform axes only)."""
    simpson_axes = scheme_options.get("simpson_axes", [0])
    if axis not in simpson_axes:
        return np.trapezoid(arr, axis=0)
    return simpson_integrate(arr, axis=0)


def _integrate_axis_newton_cotes(arr, axis, ndim, scheme_options, original_shape):
    """Composite closed Newton-Cotes (default order 4 = Boole's rule) on a unit-spacing grid."""
    nc_axes = scheme_options.get("newton_cotes_axes", [0])
    if axis not in nc_axes:
        return np.trapezoid(arr, axis=0)

    order = _get_axis_option(scheme_options.get("order", 4), axis, ndim)
    axis_size = original_shape[axis]
    w = newton_cotes_weights(axis_size, order=int(order))
    return np.tensordot(w, arr, axes=(0, 0))


def _integrate_axis_gauss_legendre(arr, axis, ndim, scheme_options, original_shape):
    """Gauss-Legendre quadrature; data must be sampled at the Gauss-Legendre nodes."""
    gauss_axes = scheme_options.get("gauss_axes", [0])
    if axis not in gauss_axes:
        return np.trapezoid(arr, axis=0)

    interval_spec = scheme_options.get("intervals", scheme_options.get("interval"))
    a, b = _get_interval(interval_spec, axis, ndim)

    axis_size = original_shape[axis]
    expected_nodes, expected_weights = gauss_legendre_nodes_and_weights(axis_size, a, b)

    nodes_spec = _get_axis_option(scheme_options.get("nodes"), axis, ndim)
    if nodes_spec is None:
        w = expected_weights
    else:
        nodes = np.asarray(nodes_spec, dtype=float)
        if nodes.shape[0] != arr.shape[0]:
            raise ValueError(
                f"Along axis {axis}, array length {arr.shape[0]} does not match "
                f"{nodes.shape[0]} supplied Gauss-Legendre nodes."
            )
        ascending_ok = np.allclose(nodes, expected_nodes, rtol=0, atol=1e-10)
        descending_ok = np.allclose(nodes, expected_nodes[::-1], rtol=0, atol=1e-10)
        if not (ascending_ok or descending_ok):
            raise ValueError(
                f"Axis {axis}: 'gauss-legendre' requires the {axis_size}-point "
                f"Gauss-Legendre node set on [{a}, {b}] (ascending or descending). "
                "Resample the data at these nodes, or use 'moment-matching' for "
                "arbitrary/scattered node sets."
            )
        w = expected_weights if ascending_ok else expected_weights[::-1]

    return np.tensordot(w, arr, axes=(0, 0))


def _integrate_axis_moment_matching(arr, axis, ndim, scheme_options, original_shape):
    """Generalized moment-matching quadrature for arbitrary/scattered 1-D node sets.

    Unlike 'truncated-cc-grid' (which defaults to a Chebyshev subgrid when no
    nodes are supplied), this scheme never infers node positions: it always
    requires the caller's true sample coordinates and the true integration
    interval, making it the right choice for irregularly/arbitrarily spaced
    or scattered data.
    """
    mm_axes = scheme_options.get("moment_matching_axes", [0])
    if axis not in mm_axes:
        return np.trapezoid(arr, axis=0)

    nodes_spec = _get_axis_option(scheme_options.get("nodes"), axis, ndim)
    if nodes_spec is None:
        raise ValueError(
            f"Axis {axis}: 'moment-matching' requires explicit per-axis 'nodes' "
            "giving the true sample coordinates; it never assumes a Chebyshev grid."
        )
    nodes = np.asarray(nodes_spec, dtype=float)
    if nodes.shape[0] != arr.shape[0]:
        raise ValueError(
            f"Along axis {axis}, array length {arr.shape[0]} does not match "
            f"{nodes.shape[0]} supplied nodes."
        )

    interval_spec = scheme_options.get("intervals", scheme_options.get("interval"))
    if interval_spec is None:
        raise ValueError(
            f"Axis {axis}: 'moment-matching' requires an explicit 'interval'/'intervals' "
            "(the true physical domain of integration); it is not assumed to be [-1, 1]."
        )
    a, b = _get_interval(interval_spec, axis, ndim)

    weight_func = _get_axis_option(scheme_options.get("weight_func"), axis, ndim)
    m = _get_axis_option(scheme_options.get("m"), axis, ndim)
    envelope_m = _get_axis_option(scheme_options.get("envelope_m"), axis, ndim)
    degree = _get_axis_option(scheme_options.get("degree"), axis, ndim)

    w = quadrature_weights(
        nodes, a, b, weight_func=weight_func, m=m, envelope_m=envelope_m, degree=degree
    )
    return np.tensordot(w, arr, axes=(0, 0))


def _integrate_axis_spline(arr, axis, ndim, scheme_options, original_shape):
    """Cubic-spline quadrature; works for uniform or arbitrary non-uniform node sets."""
    spline_axes = scheme_options.get("spline_axes", [0])
    if axis not in spline_axes:
        return np.trapezoid(arr, axis=0)

    axis_size = original_shape[axis]
    nodes_spec = _get_axis_option(scheme_options.get("nodes"), axis, ndim)
    if nodes_spec is None:
        # Fall back to the same unit-spacing index grid convention used by
        # the other schemes when no explicit sample coordinates are given.
        nodes = np.arange(axis_size, dtype=float)
    else:
        nodes = np.asarray(nodes_spec, dtype=float)
        if nodes.shape[0] != arr.shape[0]:
            raise ValueError(
                f"Along axis {axis}, array length {arr.shape[0]} does not match "
                f"{nodes.shape[0]} supplied nodes."
            )

    interval_spec = scheme_options.get("intervals", scheme_options.get("interval"))
    if interval_spec is None:
        a, b = float(nodes.min()), float(nodes.max())
    else:
        a, b = _get_interval(interval_spec, axis, ndim)

    bc_type = _get_axis_option(scheme_options.get("bc_type", "not-a-knot"), axis, ndim)
    w = spline_quadrature_weights(nodes, a, b, bc_type=bc_type)
    return np.tensordot(w, arr, axes=(0, 0))


_INTEGRATION_SCHEMES = {
    "truncated-cc-grid": _integrate_axis_truncated,
    "clenshaw-curtis": _integrate_axis_clenshaw_curtis,
    "simpson": _integrate_axis_simpson,
    "newton-cotes": _integrate_axis_newton_cotes,
    "gauss-legendre": _integrate_axis_gauss_legendre,
    "moment-matching": _integrate_axis_moment_matching,
    "cubic-spline": _integrate_axis_spline,
}


def int_arr(arr, dxs=None, scheme="trapezoidal", scheme_options=None):  # integrate an array of values on an integration domain
    # dxs is accepted for API compatibility but the grid spacing is absorbed by the weight functions
    arr = np.asarray(arr, dtype=float)
    options = {} if scheme_options is None else dict(scheme_options)
    ndim = arr.ndim
    original_shape = arr.shape
    current = arr

    axis_handler = _INTEGRATION_SCHEMES.get(scheme)
    if scheme != "trapezoidal" and axis_handler is None:
        raise ValueError(
            f"Unsupported integration scheme '{scheme}'. "
            "Supported: 'trapezoidal', 'truncated-cc-grid', 'clenshaw-curtis', "
            "'simpson', 'newton-cotes', 'gauss-legendre', 'moment-matching', 'cubic-spline'."
        )

    for axis in range(ndim):
        if scheme == "trapezoidal":
            current = np.trapezoid(current, axis=0)
        else:
            current = axis_handler(
                current, axis=axis, ndim=ndim, scheme_options=options, original_shape=original_shape
            )

    return current
