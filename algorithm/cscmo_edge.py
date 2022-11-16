import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, binary_tournament, NSGA2
from pymoo.core.mating import Mating
from pymoo.operators.selection.tournament import compare, TournamentSelection
from utils.survival import RankAndCrowdingSurvivalIgnoreConstraint
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import copy
import math
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils.SelectCand import select_exploit_explore_ind_simple,  select_cand_from_edge
from surrogate_problem.surrogate_problem import SurrogateProblem
from pymoo.algorithms.soo.nonconvex.de import Variant
from pymoo.operators.control import NoParameterControl
from algorithm.multimodal import TabuCV
from utils.help_problem import FindCvEdge
from algorithm.push_search import CCMO


class CSCMOEdge(GeneticAlgorithm):

    def __init__(self, pop_o_init, pop_size, max_FE, n_offs, niche_size, epsilon, **kwargs):
        super().__init__(pop_size=pop_size, sampling=pop_o_init,
                         advance_after_initial_infill=True,
                         n_offsprings=n_offs,
                         **kwargs)

        self.n_exploit = int(0.7 * n_offs)
        self.n_explore = n_offs - self.n_exploit
        self.survival_o = RankAndCrowdingSurvival()
        self.niche_size = niche_size
        self.cv_edge_epsilon = epsilon

    def _infill(self):
        pop_sel = None
        # print('n_gen % 2', self.n_gen % 2)
        if self.n_gen % 2 == 0:
            # push_opt_alg = CCMO(self.pop, self.pop_size, self.pop_size)
            push_opt_alg = NSGA2(pop_size=self.pop_size,sampling=self.pop)
            push_opt_alg.survival.nds = NonDominatedSorting(epsilon=0.001)
            res = minimize(self.surrogate_problem, push_opt_alg, ('n_gen', 20))
            pop_cand = res.algorithm.opt
            if pop_cand.size < self.n_exploit:
                pop_cand = self.survival_o.do(self.surrogate_problem, res.algorithm.pop, n_survive=self.n_exploit)
            pop_exploit, pop_explore = select_exploit_explore_ind_simple(pop_cand, archive=self.archive_o,
                                                                             n_exploit=self.n_exploit,
                                                                             n_explore=self.n_explore,
                                                                             mating=push_opt_alg.mating,
                                                                             problem=self.surrogate_problem,
                                                                             help_flag=False, alg=push_opt_alg)

            pop_sel = Population.merge(pop_exploit, pop_explore)

        else:
            opt_problem = FindCvEdge(self.problem, self.cv_edge_epsilon)
            pop = Population.new(X=self.pop.get('X'))
            alg = TabuCV(pop_size=self.pop_size * 5, n_offsprings=self.pop_size * 5, sampling=pop, niche_dist=self.niche_size)
            res = minimize(opt_problem, alg, ('n_gen', 50))
            pop_sel = select_cand_from_edge(res.pop, self.surrogate_problem, self.archive_o, self.n_exploit, self.n_explore)

        # print('pop_sel: ', pop_sel)
        if pop_sel is not None:
            return Population.new(X=pop_sel.get('X'))

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival_o.do(self.problem, infills)

        self.archive_o = infills
        self.surrogate_problem.fit(self.archive_o)

    def _setup(self, problem, **kwargs):
        self.surrogate_problem = SurrogateProblem(self.problem)

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self.archive_o = Population.merge(self.archive_o, infills)
            self.pop = Population.merge(self.pop, infills)
            self.pop = self.survival_o.do(self.problem, self.pop, n_survive=int(self.n_offsprings * 10))
            self.surrogate_problem.fit(self.archive_o)