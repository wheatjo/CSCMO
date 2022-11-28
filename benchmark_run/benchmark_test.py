from multiprocessing import Pool
import os
import time
import random
import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)

from pymoo.optimize import minimize
from utils.visualization import visual_process_dual_pop, visualize_process_one_pop_fix
from DisplayProblem.displaymw import *
from DisplayProblem.displayctp import *
import pickle
from utils.visualization import display_result_fix_lim, visual_process_dual_pop_fix, visualize_process_no_region
from multiprocessing import Process
from algorithm.cscmo import CSCMO
from utils.mycallback import MyCallBack
from utils.save_data import SaveData
from multiprocessing import Pool
from utils.mycallback import MyCallBack
from utils.display_indicator import MyOutput
from sample.feasible_sample import FeasibleSamplingTabu


setup = [

    {
        'problem_name': 'mw1',
        'termination': ('n_eval', 600),
        'problem': DisplayMW1(),
        'display_problem': DisplayMW1()
    },

    {
        'problem_name': 'mw2',
        'termination': ('n_eval', 600),
        'problem': DisplayMW2(),
        'niche_size': 0.5,
        'epsilon': 0.5,
        'display_problem': DisplayMW2()
    },

    {
        'problem_name': 'mw3',
        'termination': ('n_eval', 600),
        'problem': DisplayMW3(),
        'niche_size': 0.5,
        'epsilon': 0.25,
        'display_problem': DisplayMW3()
    },

    {
        'problem_name': 'mw4',
        'termination': ('n_eval', 600),
        'problem': DisplayMW4(),
        'niche_size': 0.5,
        'epsilon': 0.25,
        'display_problem': DisplayMW4()
    },

    {
        'problem_name': 'mw5',
        'termination': ('n_eval', 600),
        'problem': DisplayMW5(),
        'niche_size': 0.05,
        'epsilon': 1.5,
        'display_problem': DisplayMW5()
    },

    {
        'problem_name': 'mw6',
        'termination': ('n_eval', 600),
        'problem': DisplayMW6(),
        'display_problem': DisplayMW6()
    },

    {
        'problem_name': 'mw7',
        'termination': ('n_eval', 600),
        'problem': DisplayMW7(),
        'display_problem': DisplayMW7()
    },

    {
        'problem_name': 'mw8',
        'termination': ('n_eval', 600),
        'problem': DisplayMW8(),
        'display_problem': DisplayMW8()
    },

    {
        'problem_name': 'mw9',
        'termination': ('n_eval', 600),
        'problem': DisplayMW9(),
        'display_problem': DisplayMW9()
    },

    {
        'problem_name': 'mw10',
        'termination': ('n_eval', 600),
        'problem': DisplayMW10(),
        'display_problem': DisplayMW10()
    },

    {
        'problem_name': 'mw11',
        'termination': ('n_eval', 600),
        'problem': DisplayMW11(),
        'display_problem': DisplayMW11()
    },

    {
        'problem_name': 'mw12',
        'termination': ('n_eval', 600),
        'problem': DisplayMW12(),
        'display_problem': DisplayMW12()
    },

    {
        'problem_name': 'mw13',
        'termination': ('n_eval', 600),
        'problem': DisplayMW13(),
        'display_problem': DisplayMW13()
    },

    {
        'problem_name': 'mw14',
        'termination': ('n_eval', 600),
        'problem': DisplayMW14(),
        'display_problem': DisplayMW14()
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


def run_cscmo(config, run_number):
    problem_name = config['problem_name']
    problem = config['problem']
    pop_init = FeasibleSamplingTabu().do(problem, problem.n_var*11+5)
    print(f'--------- test {problem_name} problem  runner_num {run_number}----------------')
    algorithm = CSCMO(pop_o_init=pop_init, pop_size=100, n_offspring=10, max_eval=config['termination'][1],
                    output=MyOutput(), callback=MyCallBack())
    res = minimize(problem, algorithm, config['termination'], verbose=True, save_history=False)
    save_data_instance = SaveData(alg_name='cscmo', problem_name=problem_name,
                                  max_eval=config['termination'][1], res=res, opt_problem=config['display_problem'],
                                  benchmark_flag=False, runner_num=run_number)

    save_data_instance.process(save_ani=True, save_archive=True)


if __name__ == '__main__':

    every_problem_run_total = 11
    # process_list = []

    with Pool(processes=4) as pool:

        for i in range(len(setup)):
            test_problem_args = []

            for j in range(every_problem_run_total):
                test_problem_args.append((setup[i], j))

            iters = pool.starmap(run_cscmo, test_problem_args)
            for ret in iters:
                print(f"finish {setup[i]['problem_name']}")

    # for i in range(len(setup)):
    #     for j in range(every_problem_run_total):
    #         run_cscmo(setup[i], j)
# -----------------
    # for i in range(int(len(setup)/3)):
    #     p = Process(target=run_icnsga2, args=(setup[i], 1))
    #     p.start()
    #     process_list.append(p)
    #
    # for p in process_list:
    #     p.join()
    #
    # for i in range(int(len(setup)/3), 2*int(len(setup)/3)):
    #     p = Process(target=run_icnsga2, args=(setup[i], 1))
    #     p.start()
    #     process_list.append(p)
    # run_icnsga2(setup[0], 1)

    print('end')