"""Geometry and independent numerical checks for project-local quadrature."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import PySPIDER

import numpy as np
import pytest


def test_spatial_rules_use_their_actual_physical_nodes():
    assert (Path(__file__).parent / 'flocking_quadrature.py').is_file(), 'Quadrature-aware discrete dataset is missing'
    from flocking_quadrature import QuadratureSRDataset
    from PySPIDER.commons.library import Observable
    from PySPIDER.commons.process_library_terms import IntegrationDomain
    from PySPIDER.commons.quadrature_schemes import mapped_chebyshev_nodes, mapped_chebyshev_gauss_nodes

    positions = np.full((2, 2, 13), 7.0)
    velocities = np.ones_like(positions)
    domain = IntegrationDomain([5, 7, 0], [16, 18, 12])
    for rule, expected in [('moment-matching', np.linspace),
                           ('clenshaw-curtis', mapped_chebyshev_nodes),
                           ('chebyshev-gauss', mapped_chebyshev_gauss_nodes)]:
        data = QuadratureSRDataset(world_size=[30, 30, 13], particle_pos=positions,
            data_dict={'v': velocities}, observables=[Observable(string='v', rank=1)],
            irreps=(), kernel_sigma=1.0, cg_res=1.5, deltat=0.2,
            spatial_rule=rule, time_rule='moment-matching', moment_degree=6)
        actual = data._axis_coords(0, domain)
        a, b = 5 / 1.5, 16 / 1.5
        target = np.linspace(a, b, 12) if rule == 'moment-matching' else expected(12, a, b)
        np.testing.assert_allclose(actual, target)
        schemes, widths = data._quadrature_for_domain(domain)
        np.testing.assert_allclose(widths, [(16-5)/3, (18-7)/3, 1.2])
        np.testing.assert_allclose(schemes[0][rule]['nodes'], actual)
        np.testing.assert_allclose(data._axis_coords(2, domain), np.arange(13) * 0.2)


@pytest.mark.parametrize('rule', ['moment-matching', 'clenshaw-curtis', 'chebyshev-gauss'])
def test_kernel_is_evaluated_at_the_actual_nodes(rule):
    from flocking_quadrature import QuadratureSRDataset
    from PySPIDER.commons.library import Observable, LibraryPrime, DerivativeOrder
    from PySPIDER.discrete.library import CoarseGrainedProduct
    from PySPIDER.commons.process_library_terms import IntegrationDomain

    positions = np.repeat(np.array([[[6.0], [7.0]], [[7.0], [8.0]]]), 13, axis=2)
    data = QuadratureSRDataset(world_size=[30, 30, 13], particle_pos=positions,
        data_dict={'v': np.ones_like(positions)}, observables=[Observable(string='v', rank=1)],
        irreps=(), kernel_sigma=1.0, cg_res=1.5, deltat=0.2, rho_scale=2.0,
        spatial_rule=rule, moment_degree=6)
    domain = IntegrationDomain([5, 7, 0], [16, 18, 12])
    prime = LibraryPrime(derivand=CoarseGrainedProduct(observables=()),
                         derivative=DerivativeOrder(x_derivatives=()))
    field = data.eval_prime(prime, domain)
    x, y = data._axis_coords(0, domain), data._axis_coords(1, domain)
    X, Y = np.meshgrid(x, y, indexing='ij')
    radius_squared = 11.0
    expected = np.zeros_like(X)
    for point in positions[:, :, 0]:
        r2 = (X - point[0])**2 + (Y - point[1])**2
        expected += np.maximum(1 - r2 / radius_squared, 0)**4
    expected /= (np.pi * radius_squared / 5) * 2.0
    np.testing.assert_allclose(field, np.repeat(expected[:, :, None], 13, axis=2), atol=2e-15)


@pytest.mark.parametrize('rule', ['trapezoidal', 'moment-matching', 'clenshaw-curtis', 'chebyshev-gauss'])
def test_constant_integrals_use_the_same_physical_measure(rule):
    from flocking_quadrature import QuadratureSRDataset
    from PySPIDER.commons.process_library_terms import IntegrationDomain, Weight
    from PySPIDER.commons.library import ConstantTerm

    data = QuadratureSRDataset(world_size=[30, 30, 13], particle_pos=np.ones((2, 2, 13)),
        data_dict={'v': np.ones((2, 2, 13))}, observables=[], irreps=(),
        kernel_sigma=1.0, cg_res=1.5, deltat=0.2, spatial_rule=rule,
        time_rule='trapezoidal' if rule == 'trapezoidal' else 'moment-matching', moment_degree=6)
    domain = IntegrationDomain([5, 7, 0], [16, 18, 12])
    weight = Weight(m=[0, 0, 0], q=[0, 0, 0], k=[0, 0, 0], dxs=[1, 1, 1])
    result = data.eval_on_domain(ConstantTerm(), weight, domain)
    expected = (11/1.5) * (11/1.5) * (12*0.2)
    assert result == pytest.approx(expected, rel=5e-12)


@pytest.mark.parametrize('rule', ['moment-matching', 'clenshaw-curtis', 'chebyshev-gauss'])
def test_weighted_tensor_integral_matches_independent_gauss_legendre(rule):
    from flocking_quadrature import QuadratureSRDataset
    from PySPIDER.commons.process_library_terms import IntegrationDomain, Weight
    from numpy.polynomial import Polynomial
    from numpy.polynomial.legendre import leggauss

    data = QuadratureSRDataset(world_size=[30, 30, 13], particle_pos=np.ones((2, 2, 13)),
        data_dict={'v': np.ones((2, 2, 13))}, observables=[], irreps=(),
        kernel_sigma=1.0, cg_res=1.5, deltat=0.2, spatial_rule=rule, moment_degree=6)
    domain = IntegrationDomain([5, 7, 0], [16, 18, 12])
    _, widths = data._quadrature_for_domain(domain)
    reference_nodes = [(data._axis_coords(i, domain) - domain.min_corner[i]*data.dxs[i]) / widths[i] - 1 for i in range(3)]
    factors = [1 + c * x for c, x in zip([0.2, 0.3, 0.4], reference_nodes)]
    factors[0] = 1 + 0.2 * reference_nodes[0]**2
    field = factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :]
    data.eval_term = lambda term, dom, debug=False: field
    weight = Weight(m=[2, 2, 2], q=[1, 1, 1], k=[1, 0, 0], dxs=[1, 1, 1])
    result = data.eval_on_domain(None, weight, domain)
    s, gl_weights = leggauss(32)
    base = Polynomial([1, 0, -1])**2 * Polynomial([0, 1])
    expected = np.prod(widths)
    for i, c in enumerate([0.2, 0.3, 0.4]):
        polynomial = base.deriv() / widths[i] if i == 0 else base
        factor = 1 + c*s**2 if i == 0 else 1 + c*s
        expected *= np.dot(gl_weights, polynomial(s) * factor)
    assert abs(expected) > 1e-4
    assert result == pytest.approx(expected, rel=2e-10, abs=1e-12)


@pytest.mark.parametrize('option', [dict(wrap=True), dict(time_sigma=1), dict(spatial_rule='invalid'),
                                  dict(time_rule='clenshaw-curtis'), dict(moment_degree=-1)])
def test_rejects_unsupported_adapter_settings(option):
    from flocking_quadrature import QuadratureSRDataset

    with pytest.raises(ValueError):
        QuadratureSRDataset(world_size=[30, 30, 13], particle_pos=np.ones((2, 2, 13)),
            data_dict={'v': np.ones((2, 2, 13))}, observables=[], irreps=(),
            kernel_sigma=1.0, cg_res=1.5, deltat=0.2, **option)
