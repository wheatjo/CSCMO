import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, binary_tournament
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
from utils.SelectCand import select_exploit_explore_ind_simple, pull_stage_search, pull_stage_explore
from surrogate_problem.surrogate_problem import SurrogateProblem
from pymoo.algorithms.soo.nonconvex.de import Variant
from pymoo.operators.control import NoParameterControl
from utils.selection import BiCoSelection
from algorithm.push_search import CCMO
from algorithm.pull_search import CoStrategySearch
from utils.SelectCand import select_cand_cluster


def create_infills(pop_o_cand, pop_h_cand):
    off_o_cand = Population.new(X=pop_o_cand.get('X'))
    off_o_cand.set('archive', 'origin')
    off_h_cand = Population.new(X=pop_h_cand.get('X'))
    off_h_cand.set('archive', 'help')
    return Population.merge(off_o_cand, off_h_cand)


class CSCMO(GeneticAlgorithm):

    def __init__(self, pop_o_init, pop_size, n_offspring, **kwargs):
        super().__init__(pop_size=pop_size, sampling=pop_o_init,
                         advance_after_initial_infill=True, n_offspring=n_offspring,
                         **kwargs)
        # self.problem = problem_o
        # self.problem_help = problem_h

        self.last_gen = 2
        self.change_threshold = 1e-3
        self.push_stage = True
        self.ideal_points = [np.array([]) for i in range(self.last_gen)]
        self.nadir_points = [np.array([]) for i in range(self.last_gen)]
        self.pop_h = None
        self.max_change_pop = 10
        self.nd_sort = NonDominatedSorting()
        self.survival_o = RankAndCrowdingSurvival()
        self.survival_help = RankAndCrowdingSurvivalIgnoreConstraint()
        self.n_exploit = math.ceil(0.7 * n_offspring)
        self.n_explore = n_offspring - self.n_exploit
        self.n_cand = int(n_offspring / 2)

    def calc_max_change(self):
        delta_value = 1e-6 * np.ones(self.problem.n_obj)
        abs_ideal_point_k_1 = np.abs(self.ideal_points[0])
        abs_nadir_point_k_1 = np.abs(self.nadir_points[0])
        # delta_value = 1e-6
        rz = np.max(np.abs(self.ideal_points[-1] - self.ideal_points[0]) / np.where(abs_ideal_point_k_1 > delta_value,
                                                                                 abs_ideal_point_k_1, delta_value))
        rnk = np.max(np.abs(self.nadir_points[-1] - self.nadir_points[0]) / np.where(abs_nadir_point_k_1 > delta_value,
                                                                                 abs_nadir_point_k_1, delta_value))
        # print("[rz, rnk]: [%f, %f]", rz, rnk)

        return max(rz, rnk)

    def update_ideal_nadir(self):
        now_ideal_point = np.min(self.pop_h.get('F'), axis=0)
        now_nadir_point = np.max(self.pop_h.get('F'), axis=0)
        self.ideal_points.pop(0)
        self.ideal_points.append(now_ideal_point)
        self.nadir_points.pop(0)
        self.nadir_points.append(now_nadir_point)

    def push_pull_judge(self):
        if self.push_stage:
            self.update_ideal_nadir()

        if self.n_gen > self.last_gen and self.push_stage:
            self.max_change_pop = self.calc_max_change()
            # print("max_change", self.max_change_pop)

        if self.max_change_pop <= self.change_threshold and self.push_stage:
            self.push_stage = False

    def _infill(self):

        self.push_pull_judge()

        if self.push_stage:
            pop_init_ccmo = DefaultDuplicateElimination().do(Population.merge(self.pop, self.pop_h))
            push_opt_alg = CCMO(pop_init_ccmo, self.pop_size, self.pop_size)
            res = minimize(self.surrogate_problem, push_opt_alg, ('n_gen', 20))
            # pop_o_cand, pop_h_cand = res.algorithm.opt, res.algorithm.opt
            pop_o_cand = res.algorithm.opt
            pop_h_cand = res.algorithm.pop_h[NonDominatedSorting().do(res.algorithm.pop_h.get('F'), only_non_dominated_front=True)]

            if pop_o_cand.size < self.n_exploit:
                pop_o_cand = self.survival_o.do(self.surrogate_problem, res.algorithm.pop, n_survive=self.n_cand)

            if pop_h_cand.size < self.n_exploit:
                pop_h_cand = self.survival_help.do(self.surrogate_problem, res.algorithm.pop_h, n_survive=self.n_cand)

            pop_o_cand = select_cand_cluster(pop_o_cand, n_cand=self.n_cand, problem=self.surrogate_problem, help_flag=False)

            pop_h_cand = select_cand_cluster(pop_h_cand, n_cand=self.n_cand, problem=self.surrogate_problem, help_flag=True)

        # Pull search stage
        else:
            pop_init_costra = DefaultDuplicateElimination().do(Population.merge(self.pop, self.pop_h))
            pull_opt_alg = CoStrategySearch(pop_o_init=pop_init_costra, pop_size=self.pop_size, n_offspring=self.pop_size)
            res = minimize(self.surrogate_problem, pull_opt_alg, ('n_gen', 20))
            pop_o_cand = res.algorithm.opt
            pop_h_cand = Population.new()
            if len(res.algorithm.pop_h) > self.n_cand:
                pop_h_cand = res.algorithm.pop_h[NonDominatedSorting().do(res.algorithm.pop_h.get('F'), only_non_dominated_front=True)]

            if pop_o_cand.size < self.n_cand:
                pop_o_cand = self.survival_o.do(self.surrogate_problem, res.algorithm.pop, n_survive=self.n_cand)

            if pop_h_cand.size < self.n_cand and len(res.algorithm.pop_h):
                pop_h_cand = self.survival_help.do(self.surrogate_problem, res.algorithm.pop_h, n_survive=self.n_cand)

            pop_o_cand = select_cand_cluster(pop_o_cand, n_cand=self.n_cand, problem=self.surrogate_problem, help_flag=False)
            if len(pop_h_cand) > 0:
                pop_h_cand = select_cand_cluster(pop_h_cand, n_cand=self.n_cand, problem=self.surrogate_problem, help_flag=True)

        off_infills = create_infills(pop_o_cand, pop_h_cand)

        return off_infills

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival_o.do(self.problem, infills)
            self.pop_h = self.survival_help.do(self.problem, infills)
        self.archive_all = infills
        self.surrogate_problem.fit(self.archive_all)

    def _advance(self, infills=None, **kwargs):
        self.archive_all = Population.merge(self.archive_all, infills)
        self.pop = Population.merge(self.pop, infills)
        self.pop_h = Population.merge(self.pop_h, infills)

        self.pop = self.survival_o.do(self.problem, self.pop, n_survive=self.n_cand*20)
        self.pop_h = self.survival_help.do(self.problem, self.pop_h, n_survive=self.n_cand*20)
        self.surrogate_problem.fit(self.archive_all)

    def _setup(self, problem, **kwargs):
        self.surrogate_problem = SurrogateProblem(self.problem)
        # self.surrogate_problem.fit(self.archive_o)

    def surrogate_optimize_dual_pop(self):
        pass
