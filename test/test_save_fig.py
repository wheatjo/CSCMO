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
from utils.visualization import visual_process_dual_pop, display_result_fix_lim
from DisplayProblem.displaymw import *
from surrogate_problem.surrogate_problem import SurrogateProblem
import pickle

dir_mytest = os.path.dirname(os.path.abspath(__file__))

problem_name = 'mw1'
f = open(os.path.join(dir_mytest, 'pickle_file', 'cscmo_' + problem_name +'_data.pickle'), 'rb')
res = pickle.load(f)
save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result')
display_result_fix_lim(res.algorithm.problem, res, problem_name=problem_name, save_path=save_path)
