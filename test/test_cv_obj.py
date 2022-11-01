import sys
import os
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.constraints.as_obj import ConstraintsAsObjective
from pymoo.problems.multi.mw import *
from pymoo.optimize import minimize
from utils.visualization import visualize_process_test, display_result
from DisplayProblem.displaymw import *
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.replacement import ImprovementReplacement
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.util.misc import cdist
from pymoo.core.individual import calc_cv

class MinProblemCV(Problem):

    def __init__(self, opt_problem: Problem):
        super().__init__(n_var=opt_problem.n_var, n_obj=1, n_constr=0,
                         xl=opt_problem.xl, xu=opt_problem.xu)
        self.opt_problem = opt_problem
        self.epsilon = 0.001

    def _evaluate(self, x, out, *args, **kwargs):
        cons = self.opt_problem.evaluate(x, return_as_dictionary=True)
        cv = calc_cv(G=cons.get('G'), H=cons.get('H'))
        out['F'] = cv



class TabuCVDE(DE):

    def __init__(self, pop_size=100, n_offsprings=100, sampling=None):
        super().__init__(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling)
        self.tabu_pop_list = Population.new()
        self.min_ind_distance = 0.3

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
        I = infills.get("index")

        # replace the individuals with the corresponding parents from the mating
        self.pop[I] = ImprovementReplacement().do(self.problem, self.pop[I], infills)

        # sort the population by fitness to make the selection simpler for mating (not an actual survival, just sorting)
        self.pop = FitnessSurvival().do(self.problem, self.pop)


problem_name = 'mw3'
problem_origin = DisplayMW13()
cv_obj_problem = MinProblemCV(problem_origin)
nichGA = NicheGA(pop_size=100, norm_niche_size=0.1)
de = TabuCVDE(pop_size=100, n_offsprings=100, sampling=LatinHypercubeSampling())
res = minimize(cv_obj_problem, nichGA, ('n_gen', 100), verbose=True)
# tabu_pop = res.algorithm.tabu_pop_list
F = res.pop.get('F')
result_pop = Population.new(X=res.pop[res.pop.get('F')[:, 0] == 0].get('X'))
Evaluator().eval(problem_origin, result_pop)
display_result(problem_origin, result_pop.get('F'))


