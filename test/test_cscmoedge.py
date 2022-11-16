import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)

from algorithm.cscmo import CCMO, CSCMO
from algorithm.cscmo_edge import CSCMOEdge
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.evaluator import Evaluator
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from utils.visualization import visual_process_dual_pop, visualize_process_one_pop_fix, visualize_process_one_pop_no_region
from DisplayProblem.displaymw import *
from DisplayProblem.displayctp import *
from surrogate_problem.surrogate_problem import SurrogateProblem
import pickle
from utils.display_indicator import MyOutput
from sample.feasible_sample import FeasibleSampling, FeasibleSamplingTabu, FeasibleSamplingTabuEdge
from utils.visualization import display_result_fix_lim, visual_process_dual_pop_fix, visualize_process_no_region
from pymoo.util.display.multi import MultiObjectiveOutput


problem_name = 'mw1'
problem = DisplayMW1()
# problem = get_problem(problem_name)
output = MyOutput()

n_eval = 600
# pop_init = LatinHypercubeSampling().do(problem, problem.n_var*11)
pop_init = FeasibleSamplingTabuEdge(niche_size=0.05, epsilon=0.5).do(problem, 1000)
# pop_init = FeasibleSamplingTabu().do(problem, 1000)
# evaluator = Evaluator()
# evaluator.eval(problem, pop_init)
# termination = get_termination("n_gen", 100)

algorithm = CSCMOEdge(pop_o_init=pop_init, pop_size=problem.n_var*11, n_offs=5, max_FE=n_eval,
                      output=MultiObjectiveOutput(), niche_size=0.05, epsilon=1)

res = minimize(problem, algorithm, ('n_eval', n_eval), seed=1, verbose=True, save_history=True)

# visual_process_dual_pop(res.history, problem, problem_name+'-cscmo-surr', os.path.join(dir_mytest, 'visual_result'))
save_pickle_dir = os.path.dirname(os.path.abspath(__file__))
f = open(os.path.join(save_pickle_dir, 'pickle_file', 'cscmo_edge_'+problem_name+'_data.pickle'), 'wb')
pickle.dump(res, f)
# f = open(os.path.join(dir_mytest, 'pickle_file', 'cscmo_' + problem_name +'_data.pickle'), 'rb')
# res = pickle.load(f)

save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result')
# display_result_fix_lim(res.algorithm.problem, res, problem_name=problem_name, save_path=save_path)
# visual_process_dual_pop_fix(res.history, res.algorithm.problem, problem_name+'-cscmo-surr', save_path)
visualize_process_one_pop_fix(res.history, problem, problem_name+'-cscmo-edge', "G:/code/MyProject/CSCMO/visual_result")
# visualize_process_one_pop_no_region(res.history, problem, problem_name+'-cscmo-edge', "G:/code/MyProject/CSCMO/visual_result")