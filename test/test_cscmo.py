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
from surrogate_problem.surrogate_problem import SurrogateProblem

problem_name = 'mw3'
problem = DisplayMW3()

pop_init = LatinHypercubeSampling().do(problem, 200)
evaluator = Evaluator()
evaluator.eval(problem, pop_init)
termination = get_termination("n_gen", 100)

algorithm = CSCMO(pop_o_init=pop_init, pop_size=problem.n_var*11, n_offspring=200)
res = minimize(problem, algorithm, ('n_gen', 50), seed=1, verbose=True, save_history=True)

visual_process_dual_pop(res.history, problem, problem_name+'-cscmo-real', os.path.join(dir_mytest, 'visual_result'))

