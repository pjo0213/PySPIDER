"""Light smoke: continuous make_libraries still partitions terms by irrep."""

import numpy as np

from PySPIDER.commons.library import Observable
from PySPIDER.commons.z3base import FullRank
from PySPIDER.continuous.process_library_terms import SRDataset


def test_make_libraries_partitions_integer_irreps():
    pobs = Observable(string="p", rank=0)
    uobs = Observable(string="u", rank=1)
    data_dict = {
        "p": np.ones((6, 6, 4)),
        "u": np.ones((6, 6, 4, 2)),
    }
    srd = SRDataset(
        world_size=[6, 6, 4],
        data_dict=data_dict,
        observables=[pobs, uobs],
        dxs=[1.0, 1.0, 1.0],
        irreps=(0, 1),
    )
    srd.make_libraries(max_complexity=2, max_dt=1, max_dx=1)

    assert set(srd.libs.keys()) == {0, 1}
    assert len(srd.libs[0].terms) > 0
    assert len(srd.libs[1].terms) > 0
    assert all(term.rank == 0 for term in srd.libs[0].terms)
    assert all(term.rank == 1 for term in srd.libs[1].terms)


def test_make_libraries_partitions_fullrank_irreps():
    pobs = Observable(string="p", rank=0)
    uobs = Observable(string="u", rank=1)
    data_dict = {
        "p": np.ones((6, 6, 4)),
        "u": np.ones((6, 6, 4, 2)),
    }
    irreps = (FullRank(rank=0), FullRank(rank=1))
    srd = SRDataset(
        world_size=[6, 6, 4],
        data_dict=data_dict,
        observables=[pobs, uobs],
        dxs=[1.0, 1.0, 1.0],
        irreps=irreps,
    )
    srd.make_libraries(max_complexity=2, max_dt=1, max_dx=1)

    assert set(srd.libs.keys()) == set(irreps)
    assert all(term.rank == 0 for term in srd.libs[irreps[0]].terms)
    assert all(term.rank == 1 for term in srd.libs[irreps[1]].terms)


def test_schemes_and_options_accepts_gauss_and_lobatto_names():
    pobs = Observable(string="p", rank=0)
    srd = SRDataset(
        world_size=[4, 4, 4],
        data_dict={"p": np.ones((4, 4, 4))},
        observables=[pobs],
        dxs=[1.0, 1.0, 1.0],
        irreps=(0,),
    )
    srd.schemes_and_options = {
        0: "chebyshev-gauss",
        1: "clenshaw-curtis",
    }
    assert srd.schemes_and_options[0] == "chebyshev-gauss"
    assert srd.schemes_and_options[1] == "clenshaw-curtis"
