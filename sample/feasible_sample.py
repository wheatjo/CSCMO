from pymoo.core.sampling import Sampling
from utils.help_problem import MinProblemCV, FindCvEdge
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from utils.SelectCand import my_select_points_with_maximum_distance
from algorithm.multimodal import TabuCV
from pymoo.core.evaluator import Evaluator
from sklearn.cluster import KMeans
from pymoo.util.normalization import normalize
import numpy as np
from pysamoo.sampling.energy import EnergyConstrainedSampling
from pymoo.util.ref_dirs.energy import squared_dist, calc_potential_energy_with_grad
from pymoo.util.ref_dirs.optimizer import Adam
from pysamoo.sampling.niching import NichingConstrainedSampling
from pysamoo.sampling.rejection import RejectionConstrainedSampling
from pymoo.util.normalization import normalize, denormalize
from pymoo.util.misc import norm_eucl_dist


class FeasibleSampling(Sampling):
    
    def __init__(self):
        super(FeasibleSampling, self).__init__()
        self.max_epoch = 1

    def _do(self, problem, n_samples, **kwargs):
        opt_problem = MinProblemCV(problem)
        pop = Population.new()
        epoch = 0
        while len(pop) < n_samples:
            if len(pop) == 0:
                alg = NicheGA(pop_size=n_samples*3, norm_niche_size=0.05)
            else:
                alg = NicheGA(pop_size=n_samples*3, sampling=pop, norm_niche_size=0.05)
            res = minimize(opt_problem, alg, ('n_gen', 200), return_least_infeasible=True, verbose=True)
            pop = Population.merge(pop, res.opt[res.opt.get('F')[:, 0] <= 0])
            epoch += 1

            if len(pop) > n_samples:
                I = my_select_points_with_maximum_distance(problem, pop.get('X'), pop.get('X'), n_samples)
                pop = pop[I]

            if epoch >= self.max_epoch:
                break

        return pop.get('X')


class FeasibleSamplingTabu(Sampling):

    def __init__(self, niche_size=0.03):
        super(FeasibleSamplingTabu, self).__init__()
        self.max_epoch = 1
        self.niche_size = niche_size

    def _do(self, problem, n_samples, **kwargs):
        opt_problem = MinProblemCV(problem)
        pop = Population.new()
        alg = TabuCV(pop_size=n_samples, n_offsprings=n_samples, sampling=LatinHypercubeSampling().do(opt_problem, 1000), niche_dist=self.niche_size)
        res = minimize(opt_problem, alg, ('n_gen', 100), verbose=True)
        result_pop = Population.new(X=res.pop[res.pop.get('F')[:, 0] == 0].get('X'))
        pop_feasible = result_pop[result_pop.get('feasible')[:, 0]]
        X_norm = normalize(pop_feasible.get('X'), xl=problem.xl, xu=problem.xu)
        kmeans = KMeans(n_clusters=min(problem.n_var * 11 + 25, len(pop_feasible)), random_state=0).fit(X_norm)
        # print(len(pop_feas))
        groups = [[] for _ in range(min(problem.n_var * 11 + 25, len(pop_feasible)))]
        index = []
        for k, i in enumerate(kmeans.labels_):
            groups[i].append(k)

        for group in groups:
            if len(group) > 0:
                index.append(np.random.choice(group, 1))

        sel_ind = pop_feasible[index][:, 0]

        return sel_ind.get('X')


class MyEnergyConstrainedSampling(Sampling):

    def __init__(self,
                 func_eval_constr,
                 n_max_iter=10000):
        super().__init__()
        self.func_eval_constr = func_eval_constr
        self.n_max_iter = n_max_iter

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        constr = self.func_eval_constr
        d = problem.n_var ** 2
        pop_energy_init = kwargs['pop_init']

        if len(pop_energy_init) < n_samples:
            raise Exception('energy < n_samples !!!')

        X = pop_energy_init.get('X')

        X = normalize(X, xl, xu)

        optimizer = Adam(alpha=0.005)

        obj, grad = calc_potential_energy_with_grad(X, d)
        hist = [obj]

        done = False

        for i in range(self.n_max_iter):

            if done:
                break

            _X = optimizer.next(X, grad)
            _CV = constr(denormalize(_X, xl, xu))
            feasible = np.logical_and(_CV <= 0, np.all(np.logical_and(_X >= 0, _X <= 1), axis=1))

            X[feasible] = _X[feasible]

            obj, grad = calc_potential_energy_with_grad(X, d)
            hist.append(obj)

            hist = hist[-100:]

            avg_impr = (- np.diff(hist[-100:])).mean()

            if len(hist) > 100:
                if avg_impr < 1e-3:
                    optimizer = Adam(alpha=optimizer.alpha / 2)
                elif avg_impr < 1e-6:
                    done = True

        X = denormalize(X, xl, xu)

        return X


class FeasibleSamplingTabuEdge(Sampling):

    def __init__(self, niche_size=0.3, epsilon=0.1, pop_init=None):
        super(FeasibleSamplingTabuEdge, self).__init__()
        self.max_epoch = 1
        self.niche_size = niche_size
        self.epsilon = epsilon
        self.pop_init = pop_init

    def _do(self, problem, n_samples, **kwargs):

        while True:
            opt_problem = FindCvEdge(problem, self.epsilon)
            if self.pop_init is None:
                sampling = LatinHypercubeSampling().do(opt_problem, 100)
            else:
                sampling = self.pop_init
            alg = TabuCV(pop_size=n_samples, n_offsprings=n_samples,
                         sampling=sampling, niche_dist=self.niche_size)
            res = minimize(opt_problem, alg, ('n_gen', 100), verbose=True)
            result_pop = Population.new(X=res.pop[res.pop.get('F')[:, 0] == 0].get('X'))
            Evaluator().eval(problem, result_pop)
            if len(result_pop.get('feasible')) > int(n_samples / 5):
                break

        # pop_feasible = result_pop[result_pop.get('feasible')[:, 0]]
        pop_feasible = result_pop
        X_norm = normalize(pop_feasible.get('X'), xl=problem.xl, xu=problem.xu)
        kmeans = KMeans(n_clusters=min(problem.n_var * 11 + 25, len(pop_feasible)), random_state=0).fit(X_norm)
        # print(len(pop_feas))
        groups = [[] for _ in range(min(problem.n_var * 11 + 25, len(pop_feasible)))]
        index = []
        for k, i in enumerate(kmeans.labels_):
            groups[i].append(k)

        for group in groups:
            if len(group) > 0:
                index.append(np.random.choice(group, 1))

        sel_ind = pop_feasible[index][:, 0]

        return sel_ind.get('X')



if __name__ == '__main__':
    from pymoo.problems.multi.mw import MW2
    from DisplayProblem.displaymw import *
    from DisplayProblem.displayctp import *
    from utils.visualization import display_result
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling
    from pymoo.core.individual import calc_cv
    problem = DisplayMW2()
    # pop_init = LatinHypercubeSampling().do(problem, 500)
    pop_init = FeasibleSamplingTabuEdge(niche_size=0.5, epsilon=0.5).do(problem, 1000)
    # print(len(pop_init))
    F = problem.evaluate(pop_init.get('X'), return_values_of=['G', 'H'])
    # cv = calc_cv(F[0], F[1])
    # print(cv)
    #
    # def constr(X):
    #     F = problem.evaluate(pop_init.get('X'), return_values_of=['G', 'H'])
    #     return calc_cv(F[0], F[1])
    #
    # D = norm_eucl_dist(problem, pop_init.get('X'), pop_init.get('X'))
    # D = D.min(axis=1).argsort()
    # pop_size = problem.n_var * 11 + 24
    # index = D[-1 * pop_size:]
    # sel = pop_init[index]
    sel_F = problem.evaluate(pop_init.get('X'), return_values_of=['F'])

    # pop = MyEnergyConstrainedSampling(constr).do(problem, problem.n_var*11+24, pop_init=pop_init)

    display_result(problem, sel_F)


