import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)


from algorithm.cscmo import CCMO
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.evaluator import Evaluator
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from utils.visualization import visual_process_dual_pop, visualize_process_no_region
from DisplayProblem.displaymw import *
from surrogate_problem.surrogate_problem import SurrogateProblem
from DisplayProblem.displayctp import DisplayCTP1
from sample.feasible_simple import FeasibleSampling


problem_name = 'c2dtlz2'
# problem = DisplayMW1()
problem = get_problem(problem_name, n_obj=2)
pop_init = LatinHypercubeSampling().do(problem, 200)
# pop_init = FeasibleSampling().do(problem, 200)

evaluator = Evaluator()
evaluator.eval(problem, pop_init)
# problem_surr = SurrogateProblem(problem)
termination = get_termination("n_gen", 100)
algorithm = CCMO(pop_o_init=pop_init, pop_size=200, n_offspring=200)

res = minimize(problem, algorithm, ('n_gen', 10), seed=1, verbose=True, save_history=True)

visualize_process_no_region(res.history, problem, problem_name, "G:/code/MyProject/CSCMO/visual_result")
# print(res.pop.get('F'))
