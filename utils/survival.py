import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.normalization import normalize
from scipy.spatial.distance import cdist
from pymoo.core.individual import calc_cv

class RankAndCrowdingSurvivalIgnoreConstraint(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


class ArchSurvival(Survival):

    def __init__(self, filter_infeasible=False):
        super(ArchSurvival, self).__init__(filter_infeasible)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        # pop: P_t U A_t U Q_t
        f, g, h = pop.get('F', 'G', 'H')
        cv = calc_cv(f, g)
        f = np.column_stack([f, cv])
        nd_front = self.nds.do(f, only_non_dominated_front=True)
        nd_ind = pop[nd_front]

        archive_pop = nd_ind[~nd_ind.get('feasible').squeeze()]
        delete = self.delete(archive_pop, len(archive_pop)-n_survive)
        archive_pop = archive_pop[~(delete == 1)]
        return archive_pop

    @staticmethod
    def delete(pop_v, num_delete):
        F = pop_v.get('F')
        F_norm = normalize(F)
        cosine = 1 - cdist(F_norm, F_norm, 'cosine')
        cosine = cosine * (1 - np.eye(len(F)))
        delete_index = np.zeros(len(pop_v))
        while np.sum(delete_index) < num_delete:
            ind_min_angle_index = np.argwhere(cosine == np.max(cosine))[0]
            ind1 = pop_v[ind_min_angle_index[0]]
            ind2 = pop_v[ind_min_angle_index[1]]
            if ind1.get('CV') < ind2.get('CV') or (ind1.get('CV') == ind2.get('CV') and np.random.random() < 0.5):
                delete_index[ind_min_angle_index[1]] = 1
                cosine[:, ind_min_angle_index[1]] = 0
                cosine[ind_min_angle_index[1], :] = 0

            else:
                delete_index[ind_min_angle_index[0]] = 1
                cosine[:, ind_min_angle_index[0]] = 0
                cosine[ind_min_angle_index[0], :] = 0

        return delete_index
    