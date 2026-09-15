import numpy as np

from ..commons.process_library_terms import (
    AbstractDataset, IntegrationDomain, diff, get_slice
)
from ..commons.z3base import LiteralIndex
from .library import generate_terms_to

class SRDataset(AbstractDataset):
    # field_dict: dict[tuple[Any], np.ndarray[float]] = None 
    # storage of computed coarse-grained quantities: (prim, dims, domains) -> array
    def make_domains(self, ndomains, domain_size, pad=0):
        self.domains = []
        self.domain_size = domain_size
        for i in range(ndomains):
            min_corner = []
            max_corner = []
            for (L, max_lim) in zip(domain_size, self.world_size):
                num = np.random.randint(pad, max_lim - (L + pad) + 1)
                min_corner.append(num)
                max_corner.append(num + L - 1)
            self.domains.append(IntegrationDomain(min_corner, max_corner))
        self.pad = 0
        #return domains

    def eval_prime(self, prime, domain):
        name = prime.derivand.string
        # unpack the indices
        obs_inds = [idx.value for idx in prime.derivand.indices]
        #print(obs_inds)

        data_arr = self.data_dict[name][..., *obs_inds]
        data_slice = get_slice(data_arr, domain)

        orders = prime.derivative.get_spatial_orders()
        dimorders = [orders[LiteralIndex(i)] 
                     for i in range(self.n_dimensions-1)]
        dimorders += [prime.derivative.torder]
        #print(prime.derivative, dimorders)
        return (diff(data_slice, dimorders, self.diff_spacings(domain))
                if sum(dimorders) > 0 else data_slice)

    def generate_library_terms(self, **kwargs):
        return generate_terms_to(observables=self.observables, **kwargs)

    def find_scales(self, names=None):
        self._record_field_scales(names, check_nan=True)

    def get_char_size(self, term):
        # return characteristic size of a library term
        product = 1
        for prime in term.primes:
            xorder = prime.derivative.xorder
            torder = prime.derivative.torder
            name = prime.derivand.string
            if torder + xorder > 0:
                product *= self.scale_dict[name]['std']
            else:
                product *= self.scale_dict[name]['mean']
            product /= self.xscale ** xorder
            product /= self.tscale ** torder
        #print(f'char size of {term} is {product}')
        # if the variable is always 0 then we'll get division by zero
        return product if product > 0 else 1
