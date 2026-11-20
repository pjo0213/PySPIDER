"""Project-local discrete datasets with coordinate-correct quadrature grids.

No PySPIDER core code is replaced. Spatial kernels can be evaluated directly on
uniform, Lobatto, or Chebyshev-Gauss grids within each integration subdomain.
Temporal samples are always the original uniformly saved particle snapshots.
"""
from dataclasses import dataclass

import numpy as np

from PySPIDER.discrete.process_library_terms import SRDataset
from PySPIDER.discrete.coarse_grain_utils import poly_coarse_grain_time_slices
from PySPIDER.commons.process_library_terms import diff
from PySPIDER.commons.z3base import LiteralIndex
from PySPIDER.commons.quadrature_schemes import (
    mapped_chebyshev_nodes, mapped_chebyshev_gauss_nodes,
)


@dataclass(kw_only=True)
class QuadratureSRDataset(SRDataset):
    """Two-space-plus-time, non-wrapping, unsmoothed discrete dataset.

    ``spatial_rule``: trapezoidal, moment-matching, clenshaw-curtis, or
    chebyshev-gauss (Fejer-I integration, not a singular Chebyshev measure).
    ``time_rule``: trapezoidal or moment-matching on the saved times.
    ``moment_degree``: requested polynomial degree for moment matching; must be
    smaller than the axis sample count. Other rules do not use this option.

    Domain corners retain their original uniform-lattice/index interpretation.
    They define physical integration support endpoints; full Lobatto/Gauss
    spatial grids are generated anew on each support, not sliced from a global
    Chebyshev grid. Grids have the same sample count as the original domain.
    """
    spatial_rule: str = 'moment-matching'
    time_rule: str = 'moment-matching'
    moment_degree: int = 10

    def __post_init__(self):
        if self.spatial_rule not in {'trapezoidal', 'moment-matching', 'clenshaw-curtis', 'chebyshev-gauss'}:
            raise ValueError('Unsupported spatial quadrature rule')
        if self.time_rule not in {'trapezoidal', 'moment-matching'}:
            raise ValueError('Time rule must preserve the uniformly saved snapshots')
        if not isinstance(self.moment_degree, int) or self.moment_degree < 0:
            raise ValueError('moment_degree must be a nonnegative integer')
        if self.wrap or self.time_sigma != 0 or self.grids is not None or self.schemes_and_options:
            raise ValueError('This project adapter requires wrap=False, time_sigma=0, grids=None, and no separate schemes_and_options')
        super().__post_init__()
        if self.n_dimensions != 3:
            raise ValueError('This project adapter supports two spatial dimensions and time')

    def _axis_coords(self, axis, domain=None):
        if domain is None:
            return super()._axis_coords(axis)
        if not 0 <= axis < 3:
            raise ValueError('axis must be 0, 1, or 2')
        a = float(domain.min_corner[axis] * self.dxs[axis])
        b = float(domain.max_corner[axis] * self.dxs[axis])
        n = domain.shape[axis]
        if n < 2 or b <= a:
            raise ValueError('Each integration axis must contain at least two samples')
        if axis < 2 and self.spatial_rule == 'clenshaw-curtis':
            return mapped_chebyshev_nodes(n, a, b)
        if axis < 2 and self.spatial_rule == 'chebyshev-gauss':
            return mapped_chebyshev_gauss_nodes(n, a, b)
        return np.linspace(a, b, n)

    def _quadrature_for_domain(self, domain):
        schemes, half_widths = {}, []
        for axis in range(3):
            a = float(domain.min_corner[axis] * self.dxs[axis])
            b = float(domain.max_corner[axis] * self.dxs[axis])
            nodes = self._axis_coords(axis, domain)
            rule = self.spatial_rule if axis < 2 else self.time_rule
            half_widths.append((b - a) / 2)
            if rule == 'trapezoidal':
                schemes[axis] = rule
            else:
                options = {'nodes': nodes, 'interval': (a, b)}
                if rule == 'moment-matching':
                    if self.moment_degree >= len(nodes):
                        raise ValueError('moment_degree must be smaller than the axis sample count')
                    options['degree'] = self.moment_degree
                schemes[axis] = {rule: options}
        return schemes, half_widths

    def diff_spacings(self, domain):
        return [self._axis_coords(i, domain) for i in range(3)]

    def eval_on_domain(self, term, weight, domain, debug=False):
        """Return physical-space/time integrals for every quadrature choice.

        The core trapezoidal handler integrates with spacing one between array
        entries. Multiply by physical sample spacings on trapezoidal axes only;
        the other handlers include their interval Jacobians already.
        """
        value = super().eval_on_domain(term, weight, domain, debug)
        for axis in range(3):
            rule = self.spatial_rule if axis < 2 else self.time_rule
            if rule == 'trapezoidal':
                value *= self.dxs[axis]
        return value

    def eval_prime(self, prime, domain, experimental=True, order=4):
        """Recompute the same quartic particle kernel at the actual spatial nodes.

        No interpolation of fields or particle trajectories is used. Kernel
        normalization and support radius match discrete SRDataset's default
        experimental polynomial backend; temporal samples remain unchanged.
        """
        if not experimental or order != 4:
            raise ValueError('Only the default quartic polynomial kernel is supported')
        points = np.asarray(self.particle_pos[:, :, domain.times], dtype=np.float64)
        values = np.ones((points.shape[0], len(domain.times)), dtype=np.float64)
        for observable in prime.derivand.observables:
            indices = tuple(index.value for index in observable.indices)
            values *= self.data_dict[observable.string][:, *indices, domain.times]
        x, y = self._axis_coords(0, domain), self._axis_coords(1, domain)
        X, Y = np.meshgrid(x, y, indexing='ij')
        targets = np.column_stack([X.ravel(), Y.ravel()])
        radius = self.kernel_sigma * np.sqrt(11.0)
        field = poly_coarse_grain_time_slices(points, values, targets, order, radius)
        field = field.reshape(domain.shape) / self.rho_scale
        orders = prime.derivative.get_spatial_orders()
        derivative_orders = [orders[LiteralIndex(i)] for i in range(2)] + [prime.derivative.torder]
        if sum(derivative_orders):
            return diff(field, derivative_orders, self.diff_spacings(domain))
        return field
