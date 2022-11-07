from pymoo.core.problem import Problem
from pymoo.core.individual import calc_cv
import numpy as np

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


class ProblemIgnoreConstraint(Problem):

    def __init__(self, origin_problem):
        super().__init__(n_var=origin_problem.n_var, n_obj=origin_problem.n_obj,
                         n_ieq_constr=0, n_eq_constr=0,
                         xl=origin_problem.xl, xu=origin_problem.xu)

        self.origin_problem = origin_problem

    def _evaluate(self, x, out, *args, **kwargs):
        obj_value = self.origin_problem.evaluate(x, return_values_of=['F'])
        out['F'] = obj_value

    def _calc_pareto_front(self, *args, **kwargs):
        pf = self.origin_problem.pareto_front()
        return pf

    def get_pf_region(self):
        return self.origin_problem.get_pf_region()


class FindCvEdge(Problem):

    def __init__(self, opt_problem: Problem, epsilon: float):
        super().__init__(n_var=opt_problem.n_var, n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                         xl=opt_problem.xl, xu=opt_problem.xu)
        self.opt_problem = opt_problem
        self.epsilon = epsilon

    def _evaluate(self, x, out, *args, **kwargs):
        cons = self.opt_problem.evaluate(x, return_as_dictionary=True)

        G = cons.get('G')
        abs_G = np.abs(G)
        abs_G = np.sum(abs_G, axis=1)
        cv = abs_G - self.epsilon
        cv = np.maximum(0, cv)
        out['F'] = cv
