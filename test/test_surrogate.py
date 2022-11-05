import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)

from algorithm.cscmo import CCMO
from pymoo.algorithms.moo.nsga2 import NSGA2
from surrogate_problem.surrogate_problem import SurrogateProblem
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, binary_tournament
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.optimize import minimize
from utils.SelectCand import select_exploit_explore_ind_simple
from DisplayProblem.displaymw import *


class MOEANoConstraint(GeneticAlgorithm):

    def __init__(self, pop_o_init, pop_size, n_offspring, **kwargs):
        
        super(MOEANoConstraint, self).__init__(pop_size=pop_size, sampling=pop_o_init, n_offspring=n_offspring,
                                   output=MultiObjectiveOutput(), **kwargs)

        self.survival = RankAndCrowdingSurvival()
        self.mating_o = Mating(selection=TournamentSelection(func_comp=binary_tournament),
                               crossover=SBX(eta=15, prob=0.9),
                               mutation=PM(eta=20),
                               repair=self.repair,
                               eliminate_duplicates=self.eliminate_duplicates,
                               n_max_iterations=100)
        self.n_exploit = 3
        self.n_explore = 2

    def _infill(self):
        opt_alg = NSGA2(pop_size=200)
        res = minimize(self.surrogate_problem, opt_alg, ('n_gen', 50))
        pop_cand = res.algorithm.opt
        if pop_cand.size < self.n_exploit:
            pop_cand = self.survival.do(self.surrogate_problem, res.algorithm.pop, n_survive=self.n_exploit)

        pop_cand = self.survival.do(self.surrogate_problem, res.algorithm.pop, n_survive=self.n_exploit)
        off_exploit, off_explore = select_exploit_explore_ind_simple(pop_cand, archive=self.archive_o,
                                            n_exploit=self.n_exploit, n_explore=self.n_explore,
                                            mating=opt_alg.mating,
                                            problem=self.surrogate_problem, help_flag=True, alg=opt_alg)

        off = Population.new(X=Population.merge(off_exploit, off_explore).get('X'))
        return off

    def _initialize_advance(self, infills=None, **kwargs):
        self.archive_o = infills
        self.surrogate_problem.fit(self.archive_o)

    def _advance(self, infills=None, **kwargs):
        self.archive_o = Population.merge(self.archive_o, infills)
        self.surrogate_problem.fit(self.archive_o)

        pop = Population.merge(self.pop, infills)
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size)

    def _setup(self, problem, **kwargs):
        self.surrogate_problem = SurrogateProblem(self.problem)


def main():
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling
    from utils.help_problem import ProblemIgnoreConstraint, MinProblemCV
    from utils.visualization import visualize_process_one_pop_fix
    from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
    problem_name = 'mw1'
    problem_o = DisplayMW1()
    problem = ProblemIgnoreConstraint(problem_o)
    problem_for_cv = MinProblemCV(problem_o)
    multi_modal_cv_alg = NicheGA(pop_size=100, n_offsprings=100, sampling=LatinHypercubeSampling())
    cv_opt_res = minimize(problem_for_cv, multi_modal_cv_alg, ('n_gen', 100), verbose=True)
    pop_init = Population.new(X=cv_opt_res.pop.get('X'))
    alg = MOEANoConstraint(pop_init, pop_size=len(pop_init), n_offspring=len(pop_init))
    res = minimize(problem, alg, ('n_eval', 1000), seed=1, verbose=True, save_history=True)
    dir_save = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result', 'IgnoreConstraint')
    visualize_process_one_pop_fix(res.history, res.algorithm.problem, problem_name, dir_save)


if __name__ == '__main__':
    main()
