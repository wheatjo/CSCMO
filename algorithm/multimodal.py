from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.population import Population
import numpy as np
from pymoo.util.misc import cdist


class TabuCV(GA):

    def __init__(self, pop_size=100, n_offsprings=100, sampling=None, niche_dist=0.03):
        super().__init__(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling)
        self.tabu_pop_list = Population.new()
        self.min_ind_distance = niche_dist

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        pop_F = self.pop.get('F')
        index = np.where(pop_F == 0)[1]
        self.tabu_pop_list = Population.merge(self.tabu_pop_list, self.pop[index])

    def update_tabu_list(self, infills):
        infills_F = infills.get('F')
        index = np.where(infills_F == 0)[0]
        if len(index) == 0:
            return
        self.tabu_pop_list = Population.merge(self.tabu_pop_list, infills[index])

    def delete_infill_in_tabu(self, infills):
        X = self.tabu_pop_list.get('X')
        if len(infills) == 0 or len(self.tabu_pop_list) == 0:
            return infills
        infill_x = infills.get('X')
        D = cdist(infill_x, X)
        D[np.isnan(D)] = np.inf
        is_dup_index = np.full(len(infills), False)
        is_dup_index[np.any(D <= self.min_ind_distance, axis=1)] = True
        infills[is_dup_index].set('F', np.ones((len(infills[is_dup_index]), 1))*1e6)
        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."
        infills = self.delete_infill_in_tabu(infills)
        self.update_tabu_list(infills)
        # get the indices where each offspring is originating from
        pop = self.pop
        if infills is not None:
            pop = Population.merge(self.pop, infills)
        # sort the population by fitness to make the selection simpler for mating (not an actual survival, just sorting)
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)
