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


class CCMO(GeneticAlgorithm):
    # problem is surrogate model
    def __init__(self, pop_o_init, pop_size, n_offspring, **kwargs):
        super(CCMO, self).__init__(pop_size=pop_size, sampling=pop_o_init, n_offspring=n_offspring,
                                   output=MultiObjectiveOutput(), advance_after_initial_infill=True,**kwargs)

        self.pop_h = None
        self.survival = RankAndCrowdingSurvival()
        self.survival_help = RankAndCrowdingSurvivalIgnoreConstraint()
        self.mating_o = Mating(selection=TournamentSelection(func_comp=binary_tournament),
                               crossover=SBX(eta=15, prob=0.9),
                               mutation=PM(eta=20),
                               repair=self.repair,
                               eliminate_duplicates=self.eliminate_duplicates,
                               n_max_iterations=100)
        # mating_h
        self.mating_h = Variant(selection='rand', n_diffs=1, crossover='bin', control=NoParameterControl)
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _infill(self):
        off_o = self.mating_o.do(problem=self.problem, pop=self.pop, n_offsprings=self.n_offsprings, algorithm=self)
        off_h = self.mating_o.do(problem=self.problem, pop=self.pop_h, n_offsprings=self.n_offsprings, algorithm=self)
        self.pop = Population.merge(self.pop, off_o, off_h)
        self.pop_h = Population.merge(self.pop_h, off_o, off_h)
        self.evaluator.eval(self.problem, self.pop)
        self.evaluator.eval(self.problem, self.pop_h)

        self.pop = self.survival.do(problem=self.problem, pop=self.pop, n_survive=self.pop_size)
        self.pop_h = self.survival_help.do(problem=self.problem, pop=self.pop_h, n_survive=self.pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        # self.pop_h = Population.new(X=self.pop.get('X'), F=self.pop.get('F'))
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills)
        self.pop_h = copy.deepcopy(self.pop)
    # def _setup(self, problem, **kwargs):


class CoStrategySearch(GeneticAlgorithm):

    def __init__(self, pop_o_init, pop_size, n_offspring, epsilon_cv, **kwargs):
        super(CoStrategySearch, self).__init__(pop_size=pop_size, sampling=pop_o_init, n_offspring=n_offspring,
                                               output=MultiObjectiveOutput(), **kwargs)

        self.pop_h = None
        self.survival = RankAndCrowdingSurvival()
        self.mating_o = Mating(selection=TournamentSelection(func_comp=binary_tournament),
                               crossover=SBX(eta=15, prob=0.9),
                               mutation=PM(eta=20),
                               repair=self.repair,
                               eliminate_duplicates=self.eliminate_duplicates,
                               n_max_iterations=100)
        # mating_h
        self.survival_help = RankAndCrowdingSurvival()
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.eps_cv = epsilon_cv
        self.mating_h = Variant(selection='rand', n_diffs=1, crossover='bin', control=NoParameterControl)

    def _infill(self):
        off_o = self.mating_o.do(problem=self.problem, pop=self.pop, n_offsprings=self.n_offsprings, algorithm=self)
        off_h = self.mating_o.do(problem=self.problem, pop=self.pop_h, n_offsprings=self.n_offsprings, algorithm=self)

        off_o_copy = copy.deepcopy(off_o)
        off_h_copy = copy.deepcopy(off_h)
        self.pop = Population.merge(self.pop, off_o_copy, off_h_copy)
        self.pop_h = Population.merge(self.pop_h, off_o, off_h)
        self.evaluator.eval(self.problem, self.pop)
        self.evaluator.eval(self.problem, self.pop_h)

        cv_mat = self.pop_h.get('CV')
        cv_mat = cv_mat - self.eps_cv
        self.pop_h.set('CV', np.maximum(0, cv_mat))
        # print(f"pop_fes: \n {np.where(self.pop.get('feasible'))[0].size}")
        # print(f"pop_h_fes: \n {np.where(self.pop_h.get('feasible'))[0].size}")
        self.pop = self.survival.do(problem=self.problem, pop=self.pop, n_survive=self.pop_size)
        self.pop_h = self.survival_help.do(problem=self.problem, pop=self.pop_h, n_survive=self.pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        # self.pop_h = Population.new(X=self.pop.get('X'), F=self.pop.get('F'))
        self.pop_h = copy.deepcopy(self.pop)


def create_infills(pop_o_exploit, pop_o_explore, pop_h_exploit, pop_h_explore):
    off_o_exploit = Population.new(X=pop_o_exploit.get('X'))
    off_o_exploit.set('archive', 'origin')
    off_o_explore = Population.new(X=pop_o_explore.get('X'))
    off_o_explore.set('archive', 'origin')
    off_h_exploit = Population.new(X=pop_h_exploit.get('X'))
    off_h_exploit.set('archive', 'help')
    off_h_explore = Population.new(X=pop_h_explore.get('X'))
    off_h_explore.set('archive', 'help')
    return Population.merge(off_o_exploit, off_o_explore, off_h_exploit, off_h_explore)


class CSCMO(GeneticAlgorithm):

    def __init__(self, pop_o_init, pop_size, max_FE, **kwargs):
        super().__init__(pop_size=pop_size, sampling=pop_o_init,
                         advance_after_initial_infill=True,
                         **kwargs)
        # self.problem = problem_o
        # self.problem_help = problem_h
        self.epsilon_k = 0
        self.epsilon_0 = 0
        self.r_k = 0
        self.alpha = 0.95
        self.tao = 0.0005
        self.cp = 2
        self.Tc = 1
        self.last_gen = 4
        self.change_threshold = 1e-3
        self.max_FE = max_FE
        self.Tc = math.ceil(0.9 * math.ceil(self.max_FE / self.pop_size))
        self.push_stage = True
        self.ideal_points = [np.array([]) for i in range(self.last_gen)]
        self.nadir_points = [np.array([]) for i in range(self.last_gen)]
        self.pop_h = None
        self.max_change_pop_h = 10
        self.archive_pop = Population.new()
        self.archive_pop_h = Population.new()
        self.nd_sort = NonDominatedSorting()
        self.survival_o = RankAndCrowdingSurvival()
        self.survival_help = RankAndCrowdingSurvivalIgnoreConstraint()
        self.n_exploit = 3
        self.n_explore = 2

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

    def update_epsilon(self, epsilon_k, epsilon_0, rf, gen):
        if rf < self.alpha:
            result = (1 - self.tao) * epsilon_k

        else:
            result = epsilon_0 * (1 - (gen / self.Tc))**self.cp

        return result

    def _infill(self):

        rf_pop_h = (np.where(self.pop_h.get('feasible'))[0].size / self.pop_h.size)

        if self.push_stage:
            self.update_ideal_nadir()

        if self.n_iter > self.last_gen and self.push_stage:
            self.max_change_pop_h = self.calc_max_change()

        if self.max_change_pop_h <= self.change_threshold and self.push_stage:
            self.push_stage = False
            self.epsilon_0 = np.max(self.pop_h.get('CV'))
            self.epsilon_k = self.epsilon_0

        # !!! should not use gen, should use FE number ? change_threshold should modify !!!
        if self.push_stage is False:
            self.epsilon_k = self.update_epsilon(self.epsilon_k, self.epsilon_0, rf_pop_h, self.n_iter)

        if self.push_stage:
            pop_init_ccmo = DefaultDuplicateElimination().do(Population.merge(self.pop, self.pop_h))
            push_opt_alg = CCMO(pop_init_ccmo, self.pop_size, self.pop_size)
            res = minimize(self.surrogate_problem, push_opt_alg, ('n_gen', 100))
            pop_o_cand, pop_h_cand = res.algorithm.opt, res.algorithm.opt
            if pop_o_cand.size < self.n_exploit:
                pop_o_cand = self.survival_o.do(self.surrogate_problem, res.algorithm.pop, n_survive=self.n_exploit)

            if pop_h_cand.size < self.n_exploit:
                pop_h_cand = self.survival_help.do(self.surrogate_problem, res.algorithm.pop_h, n_survive=self.n_exploit)

            pop_o_exploit, pop_o_explore = select_exploit_explore_ind_simple(pop_o_cand, archive=self.archive_o,
                                            n_exploit=self.n_exploit, n_explore=self.n_explore,
                                            mating=push_opt_alg.mating_o,
                                            problem=self.surrogate_problem, help_flag=False, alg=push_opt_alg)

            pop_h_exploit, pop_h_explore = select_exploit_explore_ind_simple(pop_h_cand, archive=self.archive_h,
                                            n_exploit=self.n_exploit, n_explore=self.n_explore,
                                            mating=push_opt_alg.mating_o,
                                            problem=self.surrogate_problem, help_flag=True, alg=push_opt_alg)

        # Pull search stage
        else:
            pop_init_costra = DefaultDuplicateElimination().do(Population.merge(self.pop, self.pop_h))
            pull_opt_alg = CoStrategySearch(pop_o_init=pop_init_costra, pop_size=self.pop_size, n_offspring=self.pop_size,
                                            epsilon_cv=self.epsilon_k)

            # !!! warn: epsilon adaptive change, the function have not add
            res = minimize(self.surrogate_problem, pull_opt_alg, ('n_gen', 100))
            pop_o_cand, pop_h_cand = res.algorithm.opt, res.algorithm.opt
            if pop_o_cand.size < self.n_exploit:
                pop_o_cand = self.survival_o.do(self.surrogate_problem, res.algorithm.pop, n_survive=self.n_exploit)

            if pop_h_cand.size < self.n_exploit:
                pop_h_cand = self.survival_o.do(self.surrogate_problem, res.algorithm.pop_h, n_survive=self.n_exploit)

            pop_o_exploit, pop_o_explore = select_exploit_explore_ind_simple(pop_o_cand, archive=self.archive_o,
                                            n_exploit=self.n_exploit, n_explore=self.n_explore, mating=pull_opt_alg.mating_o,
                                            problem=self.surrogate_problem, help_flag=False, alg=pull_opt_alg)

            pop_h_exploit, pop_h_explore = pull_stage_search(pop_h_cand, archive=self.archive_h,
                                            n_exploit=self.n_exploit, n_explore=self.n_explore, mating=pull_opt_alg.mating_o,
                                            problem=self.surrogate_problem, help_flag=True, alg=pull_opt_alg)
            # pop_h_exploit, pop_h_explore = pull_stage_explore(res.pop, self.surrogate_problem, self.n_explore+self.n_exploit)

        off_infills = create_infills(pop_o_exploit, pop_o_explore, pop_h_exploit, pop_h_explore)
        return off_infills

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival_o.do(self.problem, infills)
        self.pop_h = self.pop
        self.archive_o = infills
        self.archive_h = copy.deepcopy(self.archive_o)
        self.archive_o.set('archive', 'origin')
        self.archive_h.set('archive', 'help')
        self.surrogate_problem.fit(self.archive_o)

    def _advance(self, infills=None, **kwargs):
        self.archive_o = Population.merge(self.archive_o, infills)
        self.archive_h = Population.merge(self.archive_h, infills[infills.get('archive') == 'help'])
        self.pop = Population.merge(self.pop, infills[infills.get('archive') == 'origin'])
        self.pop_h = Population.merge(self.pop_h, infills[infills.get('archive') == 'help'])
        self.pop = self.survival_o.do(self.problem, self.pop, n_survive=self.pop_size)
        self.pop_h = self.survival_help.do(self.problem, self.pop_h, n_survive=self.pop_size)
        self.surrogate_problem.fit(self.archive_o)

    def _setup(self, problem, **kwargs):
        self.surrogate_problem = SurrogateProblem(self.problem)
        # self.surrogate_problem.fit(self.archive_o)


    def surrogate_optimize_dual_pop(self):
        pass
