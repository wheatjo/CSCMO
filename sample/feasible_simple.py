from pymoo.core.sampling import Sampling
from utils.help_problem import MinProblemCV
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

    def __init__(self):
        super(FeasibleSamplingTabu, self).__init__()
        self.max_epoch = 1

    def _do(self, problem, n_samples, **kwargs):
        opt_problem = MinProblemCV(problem)
        pop = Population.new()
        alg = TabuCV(pop_size=500, n_offsprings=500, sampling=LatinHypercubeSampling().do(opt_problem, 500))
        res = minimize(opt_problem, alg, ('n_gen', 100), verbose=True)
        F = res.pop.get('F')
        pop_feas = Population.new(X=res.pop[res.pop.get('F')[:, 0] == 0].get('X'))
        X_norm = normalize(pop_feas.get('X'), xl=problem.xl, xu=problem.xu)
        kmeans = KMeans(n_clusters=problem.n_var * 11 + 25, random_state=0).fit(X_norm)
        # print(len(pop_feas))
        groups = [[] for _ in range(problem.n_var * 11 + 25)]
        index = []
        for k, i in enumerate(kmeans.labels_):
            groups[i].append(k)

        for group in groups:
            if len(group) > 0:
                index.append(np.random.choice(group, 1))

        sel_inds = pop_feas[index][:, 0]

        return sel_inds.get('X')

if __name__ == '__main__':
    from pymoo.problems.multi.mw import MW2
    from DisplayProblem.displaymw import *
    from DisplayProblem.displayctp import *
    from utils.visualization import display_result
    from pymoo.operators.sampling.lhs import sampling_lhs
    problem = DisplayCTP2()
    pop_init = FeasibleSamplingTabu().do(problem, 200)
    print(len(pop_init))
    F = problem.evaluate(pop_init.get('X'), return_values_of=['F'])

    display_result(problem, F)


