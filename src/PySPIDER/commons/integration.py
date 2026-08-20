# Interfacing quadrature schemes with PySPIDER

import numpy as np

from .quadrature_schemes import (
    clenshaw_curtis_weights_on_interval,
    gauss_legendre_nodes_and_weights,
    mapped_chebyshev_nodes,
    newton_cotes_weights,
    moment_matched_quad_weights,
    spline_quadrature_weights,
    truncated_chebyshev_nodes,
    _mapped_lobatto_reference_degree,
)

SUPPORTED_SCHEMES = (
    "trapezoidal",
    "newton-cotes",
    "gauss-legendre",
    "truncated-cc-grid",
    "clenshaw-curtis",
    "moment-matching",
    "cubic-spline",
)


def _interval(opts, default=None):
    spec = opts.get("interval", default)
    return float(spec[0]), float(spec[1])


def _nodes(opts):
    spec = opts.get("nodes")
    return np.asarray(spec, dtype=float)


def _dot(w, arr):
    return np.tensordot(w, arr, axes=(0, 0))


def _truncated_cc(arr, opts):
    n = arr.shape[0]
    nodes = _nodes(opts)
    a, b = _interval(opts, (-1.0, 1.0))
    if nodes is None:
        n_intervals = opts.get("num_intervals")
        if n_intervals is None:
            if not (np.isclose(a, -1.0) and np.isclose(b, 1.0)):
                raise ValueError(
                    f"truncated-cc-grid on [{a}, {b}] needs 'nodes' or 'num_intervals'"
                )
            n_intervals = n - 1
        nodes = truncated_chebyshev_nodes(int(n_intervals), a, b)
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Chebyshev nodes"
        )
    w = moment_matched_quad_weights(
        nodes, a, b,
        weight_func=opts.get("weight_func"),
        m=opts.get("m"),
        degree=opts.get("degree"),
    )
    return _dot(w, arr)


def _clenshaw_curtis(arr, opts):
    n = arr.shape[0]
    a, b = _interval(opts, (-1.0, 1.0))
    nodes = _nodes(opts)
    if nodes is None:
        n_intervals = opts.get("num_intervals", n - 1)
        nodes = mapped_chebyshev_nodes(int(n_intervals), a, b)
    elif _mapped_lobatto_reference_degree(nodes, a, b) is None:
        raise ValueError(
            f"clenshaw-curtis needs the full mapped Lobatto grid on [{a}, {b}]"
        )
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Lobatto nodes"
        )
    w = clenshaw_curtis_weights_on_interval(
        a, b, nodes.shape[0] - 1,
        m=float(opts.get("m") or 0.0),
    )
    return _dot(w, arr)


def _newton_cotes(arr, opts):
    w = newton_cotes_weights(arr.shape[0], order=int(opts.get("order", 4)))
    return _dot(w, arr)


def _gauss_legendre(arr, opts):
    n = arr.shape[0]
    a, b = _interval(opts, (-1.0, 1.0))
    expected_nodes, expected_weights = gauss_legendre_nodes_and_weights(n, a, b)
    nodes = _nodes(opts)
    if nodes is None:
        return _dot(expected_weights, arr)
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Gauss-Legendre nodes"
        )
    if np.allclose(nodes, expected_nodes, rtol=0, atol=1e-10):
        return _dot(expected_weights, arr)
    if np.allclose(nodes, expected_nodes[::-1], rtol=0, atol=1e-10):
        return _dot(expected_weights[::-1], arr)
    raise ValueError(
        f"gauss-legendre needs the {n}-point node set on [{a}, {b}]"
    )


def _moment_matching(arr, opts):
    nodes = _nodes(opts)
    a, b = _interval(opts)
    w = moment_matched_quad_weights(
        nodes, a, b,
        weight_func=opts.get("weight_func"),
        m=opts.get("m"),
        degree=opts.get("degree"),
    )
    return _dot(w, arr)


def _spline(arr, opts):
    n = arr.shape[0]
    nodes = _nodes(opts)
    if nodes is None:
        nodes = np.arange(n, dtype=float)
    elif nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} nodes"
        )
    interval = _interval(opts)
    if interval is None:
        a, b = float(nodes.min()), float(nodes.max())
    else:
        a, b = interval
    w = spline_quadrature_weights(nodes, a, b, bc_type=opts.get("bc_type", "not-a-knot"))
    return _dot(w, arr)


_HANDLERS = {
    "truncated-cc-grid": _truncated_cc,
    "clenshaw-curtis": _clenshaw_curtis,
    "newton-cotes": _newton_cotes,
    "gauss-legendre": _gauss_legendre,
    "moment-matching": _moment_matching,
    "cubic-spline": _spline,
}


def _scheme_and_opts(entry, axis):
    if not isinstance(entry, dict) or len(entry) != 1:
        raise ValueError(
            f"schemes_and_options[{axis}] must be a dict with exactly one "
            f"scheme name, got {entry!r}"
        )
    (name, opts), = entry.items()
    if opts is None:
        opts = {}
    elif not isinstance(opts, dict):
        raise ValueError(
            f"options for {name!r} on axis {axis} must be a dict, "
            f"got {type(opts).__name__}"
        )
    return name, opts


def int_arr(arr, schemes_and_options=None):
    """Integrate ``arr`` over every axis.

    ``schemes_and_options`` maps each array axis (int) to a one-entry dict
    ``{scheme_name: options_dict}``, e.g.::

        {
            0: {"newton-cotes": {"order": 4}},
            1: {"newton-cotes": {"order": 4}},
            2: {"clenshaw-curtis": {"interval": [-1, 1]}},
        }

    Omitted axes use trapezoidal. This is the same dict stored on
    ``SRDataset.schemes_and_options``. Options per scheme:

    - trapezoidal: none
    - newton-cotes: ``order`` (default: 4)
    - gauss-legendre: ``interval`` (default: [-1, 1]), ``nodes`` (default: Gauss-Legendre nodes falling within interval)
    - truncated-cc-grid: ``interval`` (default: [-1, 1]), ``nodes`` (default: truncated Chebyshev nodes on ``interval``), ``num_intervals`` (default: n-1), ``weight_func`` (default: None), ``m`` (default: None), ``degree`` (default: n-1)
    - clenshaw-curtis: ``interval`` (default: [-1, 1]), ``num_intervals`` (default: n-1), ``m`` (default: 0.0)
    - moment-matching: ``nodes`` (required), ``interval`` (required), ``weight_func`` (default: None), ``m`` (default: None), ``degree`` (default: n-1)
    - cubic-spline: ``nodes`` (default ``np.arange(n)``), ``interval`` (default [min(nodes), max(nodes)]), ``bc_type`` (default ``not-a-knot``)
    """
    arr = np.asarray(arr, dtype=float)
    if schemes_and_options is None:
        spec = {}
    elif not isinstance(schemes_and_options, dict):
        raise ValueError(
            "schemes_and_options must be a dict mapping axes to {scheme: options}"
        )
    else:
        spec = schemes_and_options
    ndim = arr.ndim
    for axis in spec:
        if not isinstance(axis, (int, np.integer)):
            raise ValueError(f"axis keys must be ints, got {axis!r}")
        if not 0 <= int(axis) < ndim:
            raise ValueError(f"axis {axis} out of range for ndim={ndim}")
    for i in range(ndim):
        if i not in spec:
            arr = np.trapezoid(arr, axis=0)
            continue
        name, opts = _scheme_and_opts(spec[i], i)
        if name == "trapezoidal":
            arr = np.trapezoid(arr, axis=0)
            continue
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ValueError(
                f"Unsupported scheme {name!r}. Supported: {', '.join(SUPPORTED_SCHEMES)}."
            )
        arr = handler(arr, opts)
    return arr
