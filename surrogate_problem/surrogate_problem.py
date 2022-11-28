from pymoo.core.problem import Problem
from pymoo.core.population import Population
import numpy as np
from scipy.interpolate import RBFInterpolator
from pymoo.core.individual import calc_cv
from surrogate.models.rbf import RBF
from surrogate.selection import ModelSelection
from surrogate.models.kriging import Kriging
from pysamoo.core.archive import Archive
import copy
from scipy.stats import norm
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


class SurrogateProblemGaussianRbf(Problem):

    def __init__(self, origin_problem: Problem, surrogate=None):
        super().__init__(origin_problem.n_var, origin_problem.n_obj, origin_problem.n_ieq_constr,
                         origin_problem.n_eq_constr,
                         origin_problem.xl, origin_problem.xu)

        # surrogate is list, [rbf_obj1, rbf_obj2, ..., rbf_constr1 ...]
        self.surrogate = surrogate if surrogate is not None else []
        self._problem = origin_problem
        self._compute_constraint_flag = True
        self.archive = None

    def fit(self, archive: Population) -> None:
        self.surrogate = np.empty((2, self.n_obj), dtype=object)
        self.archive = archive
        proto_rbf = RBF
        X, F = archive.get('X'), archive.get('F')
        for k in range(self.n_obj):
            model = ModelSelection(proto_rbf).do(X, F[:, k])
            model.fit(X, F[:, k])
            self.surrogate[0, k] = model

        proto_kriging = Kriging
        for k in range(self.n_obj):
            model = ModelSelection(proto_kriging).do(X, F[:, k])
            model.fit(X, F[:, k])
            self.surrogate[1, k] = model

    def _evaluate(self, x, out, *args, **kwargs):
        n = len(x)
        predict_rbf = np.full((n, self.n_obj), np.nan, dtype=float)
        predict_gauss = np.full((n, self.n_obj), np.nan, dtype=float)
        predict_gauss_sigma = np.full((n, self.n_obj), np.nan, dtype=float)
        for i in range(self.n_obj):
            predict_target = self.surrogate[0][i].predict(x)
            predict_rbf[:, i] = predict_target.squeeze()

        for i in range(self.n_obj):
            res = self.surrogate[1][i].predict(x, return_values_of=['y', 'sigma'])
            predict_gauss[:, i] = res[0].squeeze()
            predict_gauss_sigma[:, i] = res[1].squeeze()

        F = 0.5 * predict_rbf + 0.5 * predict_gauss - 2.0 * predict_gauss_sigma
        out['F'] = F
        res = self._problem.evaluate(x, return_as_dictionary=True)
        out['G'] = res.get('G')
        out['H'] = res.get('H')


class SurrogateProblemUCBEI(SurrogateProblemGaussianRbf):

    def _evaluate(self, x, out, ei_flag: bool, *args, **kwargs):

        if ei_flag is True:
            n = len(x)
            predict_rbf = np.full((n, self.n_obj), np.nan, dtype=float)
            predict_gauss_mu = np.full((n, self.n_obj), np.nan, dtype=float)
            predict_gauss_sigma = np.full((n, self.n_obj), np.nan, dtype=float)
            for i in range(self.n_obj):
                predict_target = self.surrogate[0][i].predict(x)
                predict_rbf[:, i] = predict_target.squeeze()

            for i in range(self.n_obj):
                # print(f'x.shape: {x.shape}, x:{x}')
                res = self.surrogate[1][i].predict(x, return_values_of=['y', 'sigma'])
                predict_gauss_mu[:, i] = res[0].squeeze()
                predict_gauss_sigma[:, i] = res[1].squeeze()

            f_min = np.min(self.archive.get('F'), axis=0)
            f = -1.0 * predict_gauss_sigma
            positive_sigma = predict_gauss_sigma > 0
            f_ei = np.full((n, self.n_obj), np.inf, dtype=float)
            for j in range(self.n_obj):
                positive_j_sigma = predict_gauss_sigma[:, j][positive_sigma[:, j]]
                mu_j = predict_gauss_mu[:, j][positive_sigma[:, j]]
                f_min_j = f_min[j]
                improve_j = f_min_j - mu_j
                z_j = improve_j / positive_j_sigma
                ei_j = improve_j * norm.cdf(z_j) + positive_j_sigma * norm.pdf(z_j)
                f_ei[:, j][positive_sigma[:, j]] = ei_j
                f_ei[:, j][~positive_sigma[:, j]] = -1 * predict_gauss_sigma[:, j][~positive_sigma[:, j]]

            F = 0.5 * predict_rbf + 0.5 * f_ei
            out['F'] = F
            res = self._problem.evaluate(x, return_as_dictionary=True)
            out['G'] = res.get('G')
            out['H'] = res.get('H')

        else:
            n = len(x)
            predict_rbf = np.full((n, self.n_obj), np.nan, dtype=float)
            predict_gauss = np.full((n, self.n_obj), np.nan, dtype=float)
            predict_gauss_sigma = np.full((n, self.n_obj), np.nan, dtype=float)
            for i in range(self.n_obj):
                predict_target = self.surrogate[0][i].predict(x)
                predict_rbf[:, i] = predict_target.squeeze()

            for i in range(self.n_obj):
                res = self.surrogate[1][i].predict(x, return_values_of=['y', 'sigma'])
                predict_gauss[:, i] = res[0].squeeze()
                predict_gauss_sigma[:, i] = res[1].squeeze()

            F = 0.5 * predict_rbf + 0.5 * predict_gauss - 3.0 * predict_gauss_sigma
            out['F'] = F
            res = self._problem.evaluate(x, return_as_dictionary=True)
            out['G'] = res.get('G')
            out['H'] = res.get('H')

            # use_sigma = predict_gauss_sigma[positive_sigma]
            #
            # improve = f_min - predict_gauss_mu[positive_sigma]
            # z = improve / use_sigma





