from pymoo.core.problem import Problem
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

