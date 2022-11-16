from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, binary_tournament
from pymoo.core.mating import Mating
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from utils.survival import ArchSurvival
from utils.selection import BiCoSelection, binary_tournament_comp_cv
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.misc import find_duplicates
from scipy.spatial.distance import cdist


def calc_crowding_distance_euclidean(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]
        mat_distance = cdist(_F, _F)
        mask = np.eye(len(mat_distance), dtype=bool).reshape(mat_distance.shape[0], -1)
        mat_dist = mat_distance[mask]
        _cd = np.sort(mat_dist, axis=1)[:, 0]
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    return crowding


class BiCoRankAndCrowdingSurvival(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=True)
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
            crowding_of_front = calc_crowding_distance_euclidean(F[front, :])

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


class CoStrategySearch(GeneticAlgorithm):

    def __init__(self, pop_o_init, pop_size, n_offspring, **kwargs):
        super(CoStrategySearch, self).__init__(pop_size=pop_size, sampling=LatinHypercubeSampling(), n_offspring=n_offspring,
                                               output=MultiObjectiveOutput(), **kwargs)

        self.pop_h = None
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.survival = RankAndCrowdingSurvival()

        self.mating = Mating(selection=TournamentSelection(func_comp=binary_tournament_comp_cv),
                             crossover=SBX(eta=15, prob=0.9),
                             mutation=PM(eta=20),
                             repair=self.repair,
                             eliminate_duplicates=self.eliminate_duplicates,
                             n_max_iterations=100)
        self.mating_h = Mating(selection=RandomSelection(),
                             crossover=SBX(eta=15, prob=0.9),
                             mutation=PM(eta=20),
                             repair=self.repair,
                             eliminate_duplicates=self.eliminate_duplicates,
                             n_max_iterations=100)

        # mating_h
        self.survival_help = ArchSurvival()
        self.termination = DefaultMultiObjectiveTermination()
        self.pre_select = BiCoSelection()

    def _infill(self):
        if len(self.pop_h) < len(self.pop):
            off = self.mating.do(self.problem, Population.merge(self.pop, self.pop_h), n_offsprings=self.pop_size)
        else:
            # print('matting_pool')
            matting_pool = self.pre_select.do(self.problem, Population.merge(self.pop, self.pop_h), len(self.pop),
                                              n_parents=1, n_pop_o=len(self.pop))
            off = self.mating_h.do(problem=self.problem, pop=matting_pool,
                                 n_offsprings=self.n_offsprings, algorithm=self)
        self.evaluator.eval(self.problem, off)
        self.pop = Population.merge(self.pop, self.pop_h, off)
        self.pop_h = Population.merge(self.pop, self.pop_h, off)
        self.pop = self.survival.do(problem=self.problem, pop=self.pop, n_survive=self.pop_size)
        self.pop_h = self.survival_help.do(problem=self.problem, pop=self.pop_h, n_survive=self.pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop_h = self.pop
