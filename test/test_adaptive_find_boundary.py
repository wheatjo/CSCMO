from sample.feasible_sample import FeasibleSamplingTabuEdge
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.visualization import display_result
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

problem_name = 'mw3'
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f = open(os.path.join(dir_mytest, 'test', 'pickle_file', 'cscmo_' + problem_name +'_data.pickle'), 'rb')
res = pickle.load(f)
pop_h = res.algorithm.pop_h
print(pop_h)
# pf_region_x, pf_region_y, pf_region_z = res.algorithm.problem.get_pf_region()
pop_infes = pop_h[~pop_h.get('feasible')[:, 0]]
pop_infes_F = pop_infes.get('F')
problem = res.algorithm.problem
find_pop = FeasibleSamplingTabuEdge(epsilon=0.2, niche_size=0.05, pop_init=Population.new(X=pop_infes.get('X'))).do(res.algorithm.problem, 100)
print(find_pop)
Evaluator().eval(problem, find_pop)
display_result(problem, find_pop.get('F'))

