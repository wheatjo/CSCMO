import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)

from algorithm.cscmo import CSCMO
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.evaluator import Evaluator
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from utils.visualization import visual_process_dual_pop, visualize_process_one_pop_fix
from DisplayProblem.displaymw import *
from DisplayProblem.displayctp import *
from surrogate_problem.surrogate_problem import SurrogateProblem
import pickle
from utils.display_indicator import MyOutput
from sample.feasible_sample import FeasibleSampling, FeasibleSamplingTabu, NichingGASampling
from utils.visualization import display_result_fix_lim, visual_process_dual_pop_fix, visualize_process_no_region, \
    visualize_process_test_3d
from pysampling.sample import sample
from pymoo.core.population import Population
from scipy.stats import qmc
from utils.save_data import SaveData
from utils.mycallback import MyCallBack

problem_name = 'mw1'
problem = DisplayMW1()
# problem = get_problem(problem_name, n_obj=2)
algorithm_name = 'cscmo'
pf = problem.pareto_front()
output = MyOutput()

n_eval = 600

pop_init = FeasibleSamplingTabu().do(problem, problem.n_var*11+5)

print(len(pop_init.get('feasible')))

print(f'test problem: {problem_name}')
algorithm = CSCMO(pop_o_init=pop_init, pop_size=100, n_offspring=10, max_eval=n_eval,
                  output=MyOutput(), callback=MyCallBack())

res = minimize(problem, algorithm, ('n_eval', n_eval), seed=1, verbose=True, save_history=False)

save_data_instance = SaveData(alg_name=algorithm_name, problem_name=problem_name, max_eval=n_eval, res=res,
                              opt_problem=problem, benchmark_flag=False)

save_data_instance.process(save_archive=True, save_ani=True)
# visual_process_dual_pop(res.history, problem, problem_name+'-cscmo-surr', os.path.join(dir_mytest, 'visual_result'))
# save_pickle_dir = os.path.dirname(os.path.abspath(__file__))
# f = open(os.path.join(save_pickle_dir, 'pickle_file/egosur', 'cscmo_'+problem_name+'_data.pickle'), 'wb')
# pickle.dump(res, f)
# f = open(os.path.join(dir_mytest, 'pickle_file', 'cscmo_' + problem_name +'_data.pickle'), 'rb')
# res = pickle.load(f)

# save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result/egosur')
# display_result_fix_lim(res.algorithm.problem, res, problem_name=problem_name, save_path=save_path)
# visual_process_dual_pop_fix(res.history, res.algorithm.problem, problem_name+'-cscmo-surr', save_path)
# visualize_process_no_region(res.history, problem, problem_name, "G:/code/MyProject/CSCMO/visual_result")
# visualize_process_one_pop_fix(res.history, res.algorithm.problem, problem_name+'-cscmoedge-surr', save_path)
# visualize_process_test_3d(res.history, problem, "mw14", "G:/code/MyProject/CSCMO/visual_result")
