# Interfacing quadrature schemes with PySPIDER

import numpy as np

from .quadrature_schemes import (
    clenshaw_curtis_weights,
    mapped_chebyshev_nodes,
    moment_matched_quad_weights,
    truncated_chebyshev_nodes,
    _ensure_weight_ready,
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


def _prepare_weight(weight, ndim):
    if weight is None:
        return None
    _ensure_weight_ready(weight)
    if len(weight.weight_objs) != ndim:
        raise ValueError(
            f"weight has {len(weight.weight_objs)} axes but array has ndim={ndim}"
        )
    return weight


def _axis_scale(weight, axis):
    if weight is None:
        return 1.0
    return weight.scale if axis == 0 else 1.0


def _trapezoidal(arr, opts, weight=None, axis=0):
    if weight is not None:
        n = arr.shape[0]
        w = np.asarray(weight.weight_objs[axis].linspace(n)[1], dtype=float)
        arr = arr * w.reshape((n,) + (1,) * (arr.ndim - 1))
    return _axis_scale(weight, axis) * np.trapezoid(arr, axis=0)


def _truncated_cc(arr, opts, weight=None, axis=0):
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
    qw = moment_matched_quad_weights(nodes, a, b, weight=weight, axis=axis)
    return _dot(_axis_scale(weight, axis) * qw, arr)


def _clenshaw_curtis(arr, opts, weight=None, axis=0):
    n = arr.shape[0]
    a, b = _interval(opts, (-1.0, 1.0))
    nodes = _nodes(opts)
    if nodes is None:
        nodes = mapped_chebyshev_nodes(n - 1, a, b)
    elif _mapped_lobatto_reference_degree(nodes, a, b) is None:
        raise ValueError(
            f"clenshaw-curtis needs the full mapped Lobatto grid on [{a}, {b}]"
        )
    if nodes.shape[0] != n:
        raise ValueError(
            f"array length {n} does not match {nodes.shape[0]} Lobatto nodes"
        )
    N = nodes.shape[0] - 1
    qw = clenshaw_curtis_weights(N, a, b, weight=weight, axis=axis)
    # Weights already match mapped_chebyshev_nodes. Flip only if the
    # caller passed descending Lobatto nodes.
    expected_asc = mapped_chebyshev_nodes(N, a, b)
    if np.allclose(nodes, expected_asc[::-1], rtol=0, atol=1e-12):
        qw = qw[::-1]
    return _dot(_axis_scale(weight, axis) * qw, arr)


def _moment_matching(arr, opts, weight=None, axis=0):
    nodes = _nodes(opts)
    if nodes is None:
        raise ValueError(
            "moment-matching requires 'nodes' and 'interval' in scheme options; "
            "for subdomain integration use truncated-cc-grid (infers nodes from "
            "array length) or pass nodes sliced to the subdomain."
        )
    n = arr.shape[0]
    if nodes.shape[0] != n:
        raise ValueError(
            f"moment-matching: array length {n} along the active axis does not "
            f"match {nodes.shape[0]} supplied nodes. Pass nodes sliced to the "
            f"integration subdomain, or use truncated-cc-grid / clenshaw-curtis "
            f"(which infer the node set from the array length)."
        )
    a, b = _interval(opts)
    qw = moment_matched_quad_weights(nodes, a, b, weight=weight, axis=axis)
    return _dot(_axis_scale(weight, axis) * qw, arr)


_HANDLERS = {
    "truncated-cc-grid": _truncated_cc,
    "clenshaw-curtis": _clenshaw_curtis,
    "moment-matching": _moment_matching,
    "trapezoidal": _trapezoidal,
}


def _scheme_and_opts(entry, axis):
    if isinstance(entry, str):
        return entry, {}
    if not isinstance(entry, dict) or len(entry) != 1:
        raise ValueError(
            f"schemes_and_options[{axis}] must be a scheme name or a dict "
            f"with exactly one scheme name, got {entry!r}"
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


def _is_legacy_dxs(arg) -> bool:
    """True for the pre-scheme second argument: a sequence of numeric spacings."""
    if arg is None or isinstance(arg, (dict, str, bytes)):
        return False
    try:
        seq = list(arg)
    except TypeError:
        return False
    if not seq:
        return False
    return all(isinstance(x, (int, float, np.integer, np.floating)) for x in seq)


def _int_arr_legacy(arr, dxs):
    """Original trapezoidal path used by older scripts: ``int_arr(arr, dxs)``."""
    if dxs is None:
        dxs = [1] * arr.ndim
    integral = np.trapezoid(arr, axis=0)
    if len(dxs) == 1:
        return integral
    return _int_arr_legacy(integral, dxs[1:])


def int_arr(arr, schemes_and_options=None, weight=None):
    """Integrate ``arr`` over every axis.

    **Modern form.** ``schemes_and_options`` maps each array axis (int) to a
    scheme name or a one-entry dict ``{scheme_name: options_dict}``, e.g.::

        {0: "clenshaw-curtis"}
        {2: {"truncated-cc-grid": {"interval": [-0.3, 0.7]}}}

    Omitted axes use trapezoidal. This is the same dict stored on
    ``AbstractDataset.schemes_and_options``. A SPIDER ``Weight`` is passed
    separately (``m``, ``q``, ``k``, ``dxs``, and the 1-D polynomials are
    read from it) and is the quadrature weight function for every scheme,
    including moment-matching. Sample count per axis is the array length.

    **Legacy form (backwards compatible).** Older scripts call
    ``int_arr(arr, dxs)`` with a sequence of numeric spacings. That path is
    still accepted: it runs composite trapezoidal in grid-index units along
    each axis and ignores the numeric values in ``dxs``. The
    weight, if any, must already be multiplied into ``arr``; do not pass
    ``weight=`` with a legacy ``dxs`` argument.

    Options are only those the dataset/Weight/array do not already
    determine:

    - trapezoidal: none (composite trapezoidal of sampled wT)
    - clenshaw-curtis: optional ``interval`` (default [-1, 1]), the
      affine image of the full Lobatto grid. A ``Weight`` stays on
      [-1, 1]; ``interval`` only supplies the Jacobian (b-a)/2. Grid
      size is n-1. For W(x)f(x) on a proper subinterval use
      truncated-cc-grid.
    - truncated-cc-grid: ``interval`` (default [-1, 1]) and either
      ``nodes`` or ``num_intervals`` of the parent Lobatto grid
    - moment-matching: ``nodes`` and ``interval`` (the arbitrary node
      set; the weight function is the ``Weight``, not a scheme option)
    """
    if _is_legacy_dxs(schemes_and_options):
        if weight is not None:
            raise ValueError(
                "legacy int_arr(arr, dxs) does not take weight=; multiply the "
                "weight into arr first, or use schemes_and_options with weight="
            )
        return _int_arr_legacy(arr, schemes_and_options)

    arr = np.asarray(arr, dtype=float)
    if schemes_and_options is None:
        spec = {}
    elif not isinstance(schemes_and_options, dict):
        raise ValueError(
            "schemes_and_options must be a dict mapping axes to "
            "a scheme name or {scheme: options}, or a legacy dxs list"
        )
    else:
        spec = schemes_and_options
    ndim = arr.ndim
    weight = _prepare_weight(weight, ndim)
    for axis in spec:
        if not isinstance(axis, (int, np.integer)):
            raise ValueError(f"axis keys must be ints, got {axis!r}")
        if not 0 <= int(axis) < ndim:
            raise ValueError(f"axis {axis} out of range for ndim={ndim}")
    for i in range(ndim):
        if i not in spec:
            name, opts = "trapezoidal", {}
        else:
            name, opts = _scheme_and_opts(spec[i], i)
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ValueError(
                f"Unsupported scheme {name!r}. Supported: {', '.join(SUPPORTED_SCHEMES)}."
            )
        arr = handler(arr, opts, weight=weight, axis=i)
    return arr