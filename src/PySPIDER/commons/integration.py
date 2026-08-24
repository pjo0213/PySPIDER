# Interfacing quadrature schemes with PySPIDER

import numpy as np

from .quadrature_schemes import (
    clenshaw_curtis_weights_on_interval,
    mapped_chebyshev_nodes,
    moment_matched_quad_weights,
    truncated_chebyshev_nodes,
    _mapped_lobatto_reference_degree,
)

SUPPORTED_SCHEMES = (
    "trapezoidal",
    "truncated-cc-grid",
    "clenshaw-curtis",
    "moment-matching",
)


def _interval(opts, default=None):
    spec = opts.get("interval", default)
    return float(spec[0]), float(spec[1])


def _nodes(opts):
    spec = opts.get("nodes")
    if spec is None:
        return None
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


_HANDLERS = {
    "truncated-cc-grid": _truncated_cc,
    "clenshaw-curtis": _clenshaw_curtis,
    "moment-matching": _moment_matching,
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
            0: {"clenshaw-curtis": {"interval": [-1, 1]}},
        }

    Omitted axes use trapezoidal. This is the same dict stored on
    ``SRDataset.schemes_and_options``. Options per scheme:

    - trapezoidal: none
    - truncated-cc-grid: ``interval`` (default: [-1, 1]), ``nodes`` (default: truncated Chebyshev nodes on ``interval``), ``num_intervals`` (default: n-1), ``weight_func`` (default: None), ``m`` (default: None), ``degree`` (default: n-1)
    - clenshaw-curtis: ``interval`` (default: [-1, 1]), ``num_intervals`` (default: n-1), ``m`` (default: 0.0)
    - moment-matching: ``nodes`` (required), ``interval`` (required), ``weight_func`` (default: None), ``m`` (default: None), ``degree`` (default: n-1)
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
