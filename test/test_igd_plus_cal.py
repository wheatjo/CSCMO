from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.evaluator import Evaluator
import numpy as np
from pymoo.visualization.scatter import Scatter
from pymoo.util.normalization import normalize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.igd import IGD

dir_point = get_reference_directions("das-dennis", 2, n_partitions=12)
print(dir_point)
problem = get_problem('mw1')
print(problem)
pop = LatinHypercubeSampling().do(problem, 50)
Evaluator().eval(problem, pop)
pop_F = pop.get('F')
nd_fronts = NonDominatedSorting().do(pop_F, only_non_dominated_front=True)
pop_nd = pop[nd_fronts]
print(len(pop_nd))
pop_F = pop_nd.get('F')
ideal = np.min(pop_F, axis=0)
nadir = np.max(pop_F, axis=0)

pop_F_normalize = normalize(pop_F, ideal, nadir)
pop_F_norm = np.sqrt(np.sum(pop_F_normalize**2, axis=1))

dot_pow = np.dot(dir_point, pop_F_normalize.T)
p_distance = np.sqrt(np.abs(dot_pow**2 - np.sum(dir_point**2, axis=1)[:, np.newaxis]))

o = np.argsort(p_distance, axis=1)[:, 0]
print(o)


def optimize_gamma(y):
    gamma = 2
    i = 0
    while True:
        if np.sum(np.power(y, gamma)) - 1 <= 1e-6:
            break
        gamma = gamma - (np.sum(np.power(y, gamma)) - 1) / np.sum(np.power(y, gamma) * np.log(y))
        i += 1
        if i >= 200:
            break

    return gamma


p = pop_nd[o][0]

get_gamma = optimize_gamma(p.get('F'))
# print(get_gamma)

# gamma_list = []
# for i in range(len(dir_point)):
#     a = pop_nd[o[i]].get('F')[np.newaxis, :]
#     b = dir_point[i][:, np.newaxis]
#     d = np.dot(a, b)
#     step_size = d / np.sqrt(np.sum(dir_point[i]**2))
#     y = step_size * dir_point[i]
#     get_gamma = optimize_gamma(y)
#     gamma_list.append(get_gamma)
#
# z = np.power(dir_point, np.array(gamma_list)[:, np.newaxis])
# if len(np.argwhere(np.isnan(z))):
#     mask = np.ones(len(z))
#     mask[np.argwhere(np.isnan(z)[:, 0])] = 0
#     z = z[mask == 1]


# print(z)

igd_ind = IGD(pop_nd.get('F'))
print(igd_ind.do(pop.get('F')))


