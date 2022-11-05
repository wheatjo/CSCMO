import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)

from algorithm.cscmo import CCMO, CSCMO
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.evaluator import Evaluator
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from utils.visualization import visual_process_dual_pop
from DisplayProblem.displaymw import *
from DisplayProblem.displayctp import *
from surrogate_problem.surrogate_problem import SurrogateProblem
import pickle

problem_name = 'ctp4'
problem = DisplayCTP4(n_var=10)
n_eval = 400
pop_init = LatinHypercubeSampling().do(problem, problem.n_var*11+25)
# evaluator = Evaluator()
# evaluator.eval(problem, pop_init)
# termination = get_termination("n_gen", 100)

algorithm = CSCMO(pop_o_init=pop_init, pop_size=problem.n_var*11, n_offspring=len(pop_init), max_FE=n_eval)
res = minimize(problem, algorithm, ('n_eval', n_eval), seed=1, verbose=True, save_history=True)

visual_process_dual_pop(res.history, problem, problem_name+'-cscmo-surr', os.path.join(dir_mytest, 'visual_result'))
save_pickle_dir = os.path.dirname(os.path.abspath(__file__))
f = open(os.path.join(save_pickle_dir, 'pickle_file', 'cscmo_'+problem_name+'_data.pickle'), 'wb')
pickle.dump(res, f)
