from pymoo.core.selection import Selection
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection, compare
import numpy as np
from pymoo.util.normalization import normalize
from scipy.spatial.distance import cdist
import math


def binary_tournament_comp_cv(pop, P):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament fro CV!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, b_cv = pop[a].CV[0], pop[b].CV[0]
        S[i] = compare(a, a_cv, b, b_cv, 'smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


class BiCoSelection(Selection):
    
    def __init__(self):
        super(BiCoSelection, self).__init__()

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        # pop: pop_o U arch
        n_pop_o = kwargs['n_pop_o']
        pop_o = pop[:n_pop_o]
        arch = pop[n_pop_o:]
        if len(arch) < n_select:
            # p = RandomSelection().do(problem, pop, n_select, n_parents=2)
            sel = TournamentSelection(func_comp=binary_tournament_comp_cv).do(problem, pop_o, n_select, n_parents=2, to_pop=False)
        else:
            sel = np.full(n_select, np.nan)
            F = pop.get('F')
            F_norm = normalize(F)
            cosine = 1 - cdist(F_norm, F_norm, 'cosine')
            # argsort return index, sort return array
            temp = np.sort(cosine, axis=1)
            k = math.ceil(math.sqrt(len(pop_o)))
            angle_density = temp[:, k]
            ad_pop = angle_density[:len(pop_o)]
            ad_arch = angle_density[len(pop_o):]
            pop_o_index = np.random.randint(0, len(pop_o), len(pop_o))
            arch_index = np.random.randint(0, len(arch), len(arch))
            i = 0
            while i < len(sel) - 1:
                if pop_o[pop_o_index[i]].CV < arch[arch_index[i]].CV:
                    sel[i] = pop_o_index[i]
                else:
                    sel[i] = len(pop_o) + arch_index[i] - 1

                if ad_pop[pop_o_index[i+1]] > ad_arch[arch_index[i+1]]:
                    sel[i+1] = pop_o_index[i+1]
                else:
                    sel[i+1] = len(pop_o) + arch_index[i+1] - 1

                i += 2
        return sel.astype(int, copy=False)


if __name__ == '__main__':
    from pymoo.problems.multi.mw import MW3
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population
    import pickle
    from utils.visualization import display_result
    import os

    problem_name = 'mw3'
    dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = open(os.path.join(dir_mytest, 'test', 'pickle_file', 'cscmo_' + problem_name + '_data.pickle'), 'rb')
    res = pickle.load(f)
    problem = res.algorithm.problem
    pop_o = res.algorithm.pop
    arch = res.algorithm.pop_h
    sel_ind = BiCoSelection().do(problem, Population.merge(pop_o, pop_o), n_select=len(pop_o), n_parents=1, n_pop_o=len(pop_o))
    display_result(problem, sel_ind.get('F'))
    display_result(problem, pop_o.get('F'))





