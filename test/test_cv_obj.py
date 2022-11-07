import sys
import os
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.constraints.as_obj import ConstraintsAsObjective
from pymoo.problems.multi.mw import *
from pymoo.optimize import minimize
from utils.visualization import display_result
from DisplayProblem.displaymw import *
from DisplayProblem.displayctp import *
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.replacement import ImprovementReplacement
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.util.misc import cdist
from pymoo.core.individual import calc_cv
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from sklearn.cluster import KMeans
from pymoo.util.normalization import normalize

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


class FindCvEdge(Problem):

    def __init__(self, opt_problem: Problem):
        super().__init__(n_var=opt_problem.n_var, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                         xl=opt_problem.xl, xu=opt_problem.xu)
        self.opt_problem = opt_problem
        self.epsilon = 1.5

    def _evaluate(self, x, out, *args, **kwargs):
        cons = self.opt_problem.evaluate(x, return_as_dictionary=True)

        G = cons.get('G')
        abs_G = np.abs(G)
        abs_G = np.sum(abs_G, axis=1)
        cv = abs_G - self.epsilon
        cv = np.maximum(0, cv)
        out['F'] = cv


class TabuCV(GA):

    def __init__(self, pop_size=100, n_offsprings=100, sampling=None):
        super().__init__(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling)
        self.tabu_pop_list = Population.new()
        self.min_ind_distance = 0.01

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


problem_name = 'mw3'
problem_origin = DisplayCTP8()

cv_obj_problem = MinProblemCV(problem_origin)
nichGA = NicheGA(pop_size=100, norm_niche_size=0.05)
X = LatinHypercubeSampling().do(cv_obj_problem, 100).get('X')
cv_obj_problem.evaluate(X)
de = TabuCV(pop_size=500, n_offsprings=500, sampling=LatinHypercubeSampling().do(cv_obj_problem, 100))
res = minimize(cv_obj_problem, de, ('n_gen', 100), verbose=True)
# tabu_pop = res.algorithm.tabu_pop_list
F = res.pop.get('F')
result_pop = Population.new(X=res.pop[res.pop.get('F')[:, 0] == 0].get('X'))
Evaluator().eval(problem_origin, result_pop)
AA = result_pop.get('feasible')
pop_feas = result_pop[result_pop.get('feasible')[:, 0]]
print(len(pop_feas))
X_norm = normalize(pop_feas.get('X'), xl=problem_origin.xl, xu=problem_origin.xu)
kmeans = KMeans(n_clusters=problem_origin.n_var*11+25, random_state=0).fit(X_norm)
# print(len(pop_feas))
groups = [[] for _ in range(problem_origin.n_var*11+25)]
index = []
for k, i in enumerate(kmeans.labels_):
    groups[i].append(k)

for group in groups:
    if len(group) > 0:
        index.append(np.random.choice(group, 1))

sel_inds = pop_feas[index][:, 0]
if len(result_pop) == 0:
    print('no find')
elif len(pop_feas) == 0:
    display_result(problem_origin, result_pop.get('F'))
else:
    display_result(problem_origin, sel_inds.get('F'))
    # display_result(problem_origin, result_pop.get('F'))




