import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, binary_tournament
from pymoo.core.mating import Mating
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import copy
from pymoo.algorithms.soo.nonconvex.de import Variant
from pymoo.operators.control import NoParameterControl
from utils.survival import ArchSurvival
from utils.selection import BiCoSelection
from pymoo.operators.sampling.lhs import LatinHypercubeSampling


class CoStrategySearch(GeneticAlgorithm):

    def __init__(self, pop_o_init, pop_size, n_offspring, **kwargs):
        super(CoStrategySearch, self).__init__(pop_size=pop_size, sampling=LatinHypercubeSampling(), n_offspring=n_offspring,
                                               output=MultiObjectiveOutput(), **kwargs)

        self.pop_h = None
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.survival = RankAndCrowdingSurvival()

        self.mating = Mating(selection=BiCoSelection(),
                             crossover=SBX(eta=15, prob=0.9),
                             mutation=PM(eta=20),
                             repair=self.repair,
                             eliminate_duplicates=self.eliminate_duplicates,
                             n_max_iterations=100)
        # mating_h
        self.survival_help = ArchSurvival()
        self.termination = DefaultMultiObjectiveTermination()

    def _infill(self):
        off = self.mating.do(problem=self.problem, pop=Population.merge(self.pop, self.arch),
                             n_offsprings=self.n_offsprings, algorithm=self)
        self.pop = Population.merge(self.pop, off)
        self.arch = Population.merge(self.pop, self.arch, off)
        self.evaluator.eval(self.problem, self.pop)
        self.evaluator.eval(self.problem, self.pop_h)

        # print(f"pop_fes: \n {np.where(self.pop.get('feasible'))[0].size}")
        # print(f"pop_h_fes: \n {np.where(self.pop_h.get('feasible'))[0].size}")
        self.pop = self.survival.do(problem=self.problem, pop=self.pop, n_survive=self.pop_size)
        self.arch = self.survival_help.do(problem=self.problem, pop=self.arch, n_survive=self.pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        self.arch = self.pop


if __name__ == '__main__':
    import pickle
    import os
    problem_name = 'mw3'
    dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = open(os.path.join(dir_mytest, 'test', 'pickle_file', 'cscmo_' + problem_name + '_data.pickle'), 'rb')
    res = pickle.load(f)
