from multiprocessing import Pool
import os
import time
import random


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
from sample.feasible_sample import FeasibleSampling, FeasibleSamplingTabu
from utils.visualization import display_result_fix_lim, visual_process_dual_pop_fix, visualize_process_no_region
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.algorithms.moo.nsga2 import NSGA2
from multiprocessing import  Process

setup = [

    {
        'problem_name': 'mw1',
        'termination': ('n_eval', 600),
        'problem': DisplayMW1()
    },

    {
        'problem_name': 'mw2',
        'termination': ('n_eval', 600),
        'problem': DisplayMW2(),
        'niche_size': 0.5,
        'epsilon': 0.5
    },

    {
        'problem_name': 'mw3',
        'termination': ('n_eval', 600),
        'problem': DisplayMW3(),
        'niche_size': 0.5,
        'epsilon': 0.25
    },


    {
        'problem_name': 'mw5',
        'termination': ('n_eval', 600),
        'problem': DisplayMW5(),
        'niche_size': 0.05,
        'epsilon': 1.5
    },

    {
        'problem_name': 'mw6',
        'termination': ('n_eval', 600),
        'problem': DisplayMW6()
    },

    {
        'problem_name': 'mw7',
        'termination': ('n_eval', 600),
        'problem': DisplayMW7()
    },

    {
        'problem_name': 'mw9',
        'termination': ('n_eval', 600),
        'problem': DisplayMW9()
    },

    {
        'problem_name': 'mw10',
        'termination': ('n_eval', 600),
        'problem': DisplayMW10()
    },

    {
        'problem_name': 'mw11',
        'termination': ('n_eval', 600),
        'problem': DisplayMW10()
    },

    {
        'problem_name': 'mw13',
        'termination': ('n_eval', 600),
        'problem': DisplayMW13()
    },
]


# def run_algorithm(config, run_number):
#     problem_name = config['problem_name']
#     problem = config['problem']
#     # pop_init = LatinHypercubeSampling().do(problem, problem.n_var*11)
#     algorithm = NSGA2(pop_size=200)
#     res = minimize(problem, algorithm, config['termination'], seed=1, verbose=False, save_history=True)
#     save_pickle_dir = os.path.dirname(os.path.abspath(__file__))
#     f = open(os.path.join(save_pickle_dir, 'pickle_file', 'runner', f"nsga2_{problem_name}_data_{run_number} .pickle"), 'wb')
#     pickle.dump(res, f)


def run_cscmo_edge(config, run_number):
    problem_name = config['problem_name']
    problem = config['problem']
    pop_init = LatinHypercubeSampling().do(problem, problem.n_var*11)
    algorithm = CSCMOEdge(pop_o_init=pop_init, pop_size=problem.n_var * 11, n_offs=5, max_FE=config['termination'][1],
                          output=MultiObjectiveOutput(), niche_size=0.05, epsilon=1)

    res = minimize(problem, algorithm, config['termination'], seed=1, verbose=False, save_history=True)

    save_pickle_dir = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(save_pickle_dir, 'pickle_file', 'runner', f"cscmo_edge_{problem_name}_data_{run_number} .pickle"), 'wb')
    pickle.dump(res, f)


if __name__ == '__main__':

    every_problem_run_total = 11
    process_list = []
    # for i in range(len(setup)):  # 开启5个子进程执行fun1函数
    #     for j in range(every_problem_run_total):
    #         p = Process(target=run_cscmo_edge, args=(setup[i], j))  # 实例化进程对象
    #         p.start()
    #         process_list.append(p)
    #
    #     for p in process_list:
    #         p.join()

    for i in range(len(setup)):
        p = Process(target=run_cscmo_edge, args=(setup[i], 1))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print('end')



