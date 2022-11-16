from pymoo.core.problem import Problem
from pymoo.core.population import Population
import numpy as np
from scipy.interpolate import RBFInterpolator
from pymoo.core.individual import calc_cv
from surrogate.models.rbf import RBF
from surrogate.selection import ModelSelection

# suppose surrogate problem just for expensive object and cheap constraint
# only build surrogate model for object function
class SurrogateProblem(Problem):

    def __init__(self, origin_problem: Problem, surrogate=None):
        super().__init__(origin_problem.n_var, origin_problem.n_obj, origin_problem.n_ieq_constr,
                         origin_problem.n_eq_constr,
                         origin_problem.xl, origin_problem.xu)

        # surrogate is list, [rbf_obj1, rbf_obj2, ..., rbf_constr1 ...]
        self.surrogate = surrogate if surrogate is not None else []
        self._problem = origin_problem
        self._compute_constraint_flag = True

    def fit(self, archive: Population) -> None:
        self.surrogate = []

        # fit_data_pop = Population.new(X=archive.get('X'), F=archive.get('F'))
        # X = fit_data_pop.get('X')
        # x_unique_index = np.unique(X, return_index=True, axis=0)[1]
        # fit_data_pop_unique = fit_data_pop[x_unique_index]
        # # print(f"X shape:{len(fit_data_pop_unique)}")
        # # print(f"objs shape:{len(fit_data_pop_unique)}")
        # X = fit_data_pop_unique.get('X')
        # objs = fit_data_pop_unique.get('F')
        # X, objs, constrains shape should equal 2
        # for i in range(self.n_obj):
        #
        #     target = RBFInterpolator(X, objs[:, i][:, np.newaxis])
        #     # target = RBFInterpolator(X, objs[:, i])
        #     self.surrogate.append(target)
        proto = RBF
        X, F = archive.get('X'), archive.get('F')
        for k in range(self.n_obj):
            model = ModelSelection(proto).do(X, F[:, k])
            model.fit(X, F[:, k])
            self.surrogate.append(model)


    def _evaluate(self, x, out, *args, **kwargs):
        n = len(x)
        predict = np.full((n, self.n_obj), np.nan, dtype=float)
        for i in range(self.n_obj):
            predict_target = self.surrogate[i].predict(x)
            predict[:, i] = predict_target.squeeze()

        out['F'] = predict
        res = self._problem.evaluate(x, return_as_dictionary=True)
        out['G'] = res.get('G')
        out['H'] = res.get('H')
