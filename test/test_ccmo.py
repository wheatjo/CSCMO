import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)


from algorithm.pull_search import CoStrategySearch
from algorithm.push_search import CCMO
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.evaluator import Evaluator
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from utils.visualization import visual_process_dual_pop, visualize_process_no_region, visual_process_dual_pop_fix
from DisplayProblem.displaymw import *
from DisplayProblem.displayctp import *
from surrogate_problem.surrogate_problem import SurrogateProblem
from DisplayProblem.displayctp import DisplayCTP1
from sample.feasible_sample import FeasibleSampling
from utils.mycallback import MyCallBack
from utils.save_data import SaveData


problem_name = 'mw13'
alg_name = 'ccmo'
problem = DisplayMW13()
# problem = get_problem(problem_name, n_obj=2)
pop_init = LatinHypercubeSampling().do(problem, 100)
# pop_init = FeasibleSampling().do(problem, 200)

evaluator = Evaluator()
evaluator.eval(problem, pop_init)
# problem_surr = SurrogateProblem(problem)
termination = get_termination("n_eval", 30000)
algorithm = CCMO(pop_o_init=pop_init, pop_size=100, n_offspring=100)

res = minimize(problem, algorithm, termination, seed=1, verbose=True, save_history=False, callback=MyCallBack())
save_data_instance = SaveData(alg_name, problem_name, 30000, res)
save_data_instance.process(save_ani=True, save_archive=True)

# save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result/defw2')
# visual_process_dual_pop_fix(res.history, res.algorithm.problem, problem_name+'-Cos-100', save_path)
# visualize_process_no_region(res.history, problem, problem_name, "G:/code/MyProject/CSCMO/visual_result")
# print(res.pop.get('F'))
