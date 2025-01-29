# It may or may not be nicer to take the SRDataset object as input for some of these
from timeit import default_timer as timer
from functools import reduce
from operator import add

from library import *
from commons.sparse_reg import *
from commons.sparse_reg_bf import *
    
def identify_equations(lib_object, reg_opts, print_opts=None, threshold=1e-5, min_complexity=1, # ranks=None,
                       max_complexity=None, max_equations=999, timed=True, experimental=True,
                       excluded_terms=None, primes=None):
    if timed:
        start = timer()
    equations = []
    lambdas = []
    reg_results = []
    derived_eqns = {}

    library = lib_object.terms
    Q = lib_object.Q
    
    #if ranks is None:
    #    ranks = (0, 1, 2)
    if print_opts is None:
        print_opts = {'num_format': '{0:.3g}', 'latex_output': False}
    if excluded_terms is None:
        excluded_terms = set()
    # this can be eliminated by keeping track of two different max_complexities in args
    lib_max_complexity = max([term.complexity for term in library])  # generate list of derived terms up to here
    if max_complexity is None:
        max_complexity = int(np.ceil(lib_max_complexity))
    if primes is None:
        primes = get_primes(library, max_complexity)
    for complexity in range(min_complexity, max_complexity + 1):
        while len(equations) < max_equations:
            selection = [(term, i) for (i, term) in enumerate(library) if term.complexity <= complexity
                         and term not in excluded_terms]
            if len(selection) == 0:  # no valid terms of this complexity
                break
            sublibrary = [s[0] for s in selection]
            inds = [s[1] for s in selection]
            
            # identify model
            if experimental:
                # only difference in bf regression conventions is that Xi corresponds to full library
                #print(reg_opts['scaler'], '; sub_inds:', inds, '; full_cs:', reg_opts['scaler'].full_cs)
                reg_opts['scaler'].reset_inds(inds)
                #print(reg_opts)
                eq, res, reg_result = make_equation_from_Xi(sparse_reg_bf(Q, **reg_opts), library, threshold)
            else:
                reg_opts['subinds'] = inds
                eq, res, reg_result = make_equation_from_Xi(sparse_reg(Q, **reg_opts), sublibrary, threshold)
            reg_result.sublibrary = sublibrary # record what the terms actually are
            if 'verbose' in reg_opts.keys() and reg_opts['verbose']:
                print('Result:', eq, '. residual:', res)
            if res > threshold:
                break
            equations.append(eq)
            lambdas.append(res)
            reg_results.append(reg_result)
            # add some output about the discovered model
            if timed:
                # noinspection PyUnboundLocalVariable
                time = timer() - start
                print(f"[{time:.2f} s]")
            print(f'Identified model: {eq.pstr(**print_opts)} (order {complexity}, residual {res:.2e})')
            # eliminate terms via infer_equations
            derived_eqns[str(eq)] = []
            for new_eq in infer_equations(eq, primes, lib_max_complexity):
                #print("NEW_EQ:", new_eq)
                lhs, rhs = new_eq.eliminate_complex_term()
                #if 'verbose' in reg_opts.keys() and reg_opts['verbose']:
                #    print("Inferred equation:", new_eq)
                #    print("Excluded term:", lhs)
                excluded_terms.add(lhs)
                #for t in excluded_terms:
                #    print(f"{lhs} =? {t}:", lhs==t)
                #if 'verbose' in reg_opts.keys() and reg_opts['verbose']:
                #    print("All excluded terms so far:", excluded_terms)
                derived_eqns[str(eq)].append(form_equation(lhs, rhs))
    return equations, lambdas, reg_results, derived_eqns, excluded_terms

def interleave_identify(lib_objects, reg_opts_list, print_opts=None, threshold=1e-5, min_complexity=1,  # ranks = None
                        max_complexity=None, max_equations=999, timed=True, experimental=True,
                        excluded_terms=None):
    equations = []
    lambdas = []
    reg_results = []
    derived_eqns = {}
    #if ranks is None:
    #    ranks = (0, 1, 2)
    libraries = [lib_object.terms for lib_object in lib_objects]
    
    if excluded_terms is None:
        excluded_terms = set()
    if max_complexity is None:
        max_complexity = int(np.ceil(max([term.complexity for library in libraries for term in library])))
    concat_libs = reduce(add, libraries, [])
    primes = get_primes(concat_libs, max_complexity)
    for complexity in range(min_complexity, max_complexity + 1):
        for lib_object, reg_opts in zip(lib_objects, reg_opts_list):
            #if 'verbose' in reg_opts.keys() and reg_opts['verbose']:
            print("--- WORKING ON LIBRARY WITH IRREP", lib_object.irrep, "AT COMPLEXITY", complexity, '---')
            #print("Symmetry:", translate_symmetry(library[0].symmetry()))
            eqs_i, lbds_i, rrs_i, der_eqns_i, exc_terms_i = identify_equations(lib_object, reg_opts,
                                                                        threshold=threshold,
                                                                        min_complexity=complexity,
                                                                        max_complexity=complexity,
                                                                        max_equations=max_equations, timed=timed,
                                                                        excluded_terms=excluded_terms,
                                                                        experimental=experimental, primes=primes)
        
            equations += eqs_i
            lambdas += lbds_i
            reg_results += rrs_i
            match lib_object.irrep:
                case int() | FullRank(): # these implications are always true
                    derived_eqns.update(der_eqns_i)
                    excluded_terms.update(exc_terms_i)
                case Antisymmetric() | SymmetricTraceFree():
                    pass # these implications depend on the specific irrep's symmetry and shouldn't be reused
    return equations, lambdas, reg_results, derived_eqns, excluded_terms

def make_equation_from_Xi(reg_result, sublibrary, threshold):
    Xi = reg_result.xi
    lambd = reg_result.lambd
    best_term = reg_result.best_term
    lambda1 = reg_result.lambda1 
    if lambda1 < lambd or lambda1 < threshold: # always select sub-threshold one-term model
        return Equation(terms=(sublibrary[best_term],), coeffs=(1,)), lambda1, reg_result
    else:
        zipped = [(sublibrary[i], c) for i, c in enumerate(Xi) if c != 0]
        return Equation(terms=[e[0] for e in zipped], coeffs=[e[1] for e in zipped]).canonicalize(), lambd, reg_result

def infer_equations(equation, primes, max_complexity, complexity=None):
    if complexity is None:
        complexity = max([term.complexity for term in equation.terms])
    if complexity > max_complexity:
        return
    #print('eq:', equation, 'rank:', equation.rank, 'primes:', primes, 'max_c:', max_complexity, 'c:', complexity)
    # do all of the contractions in one step so we don't have different permutations of contraction & index creation 
    yield from get_all_contractions(equation)

    if complexity == max_complexity:
        return
    #print('eq_dt & eq_dx:')
    eq_dt = dt_fun(equation)#.canonicalize() # I don't think canonicalization is necessary here
    eq_dx = dx_fun(equation)#.canonicalize()
    #print(eq_dt, '&', eq_dx)
    yield from infer_equations(eq_dt, primes, max_complexity, complexity=complexity+1)
    yield from infer_equations(eq_dx, primes, max_complexity, complexity=complexity+1)

    rem_complexity = max_complexity - complexity
    for prime in primes:
        # RESTORE
        if prime.complexity <= rem_complexity:
            #print('prime', prime)
            #print('prime*eq', prime * equation, 'new_comp', complexity+prime.complexity)
            yield from infer_equations(prime * equation, primes, max_complexity,
                                   complexity=complexity+prime.complexity) 

def get_all_contractions(equation):
    #print('Equation', equation)
    ce = canonicalize(equation)
    #print("Canonicalized:", ce)
    yield ce # base case
    for i in range(equation.rank):
        for j in range(i+1, equation.rank):
            #print('Contracting', i, j)
            yield from get_all_contractions(contract(equation, i, j))

def form_equation(lhs, rhs):
    if rhs is None:
        return Equation(terms=(lhs,), coeffs=(1,))
    else:
        return Equation(terms=(lhs,)+rhs.terms, coeffs=(1,) + tuple([-c for c in rhs.coeffs])).canonicalize()

def get_primes(library, max_complexity):
    all_primes = set(prime.purge_indices() for term in library 
                     for prime in term.primes if prime.complexity<=max_complexity)
    return all_primes