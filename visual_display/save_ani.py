import pickle
from utils.visualization import display_result, visual_process_dual_pop_fix, visualize_process_one_pop_fix
import os
from pymoo.visualization.scatter import Scatter
from pymoo.util.optimum import filter_optimum
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
problem_name = 'mw1'
f = open(os.path.join(dir_mytest, 'test/pickle_file/BiCo', 'cscmo_'+problem_name+'_data.pickle'), 'rb')

# problem_name_list = ['mw5', 'mw6', 'mw7', 'mw9', 'mw10', 'mw11', 'mw12', 'mw13']
# for problem_name in problem_name_list:
# f = open(os.path.join(dir_mytest, 'test', 'pickle_file', 'cscmo_' + problem_name +'_data.pickle'), 'rb')
res = pickle.load(f)
archive_opt = filter_optimum(res.algorithm.archive_o)

dir_save = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result/BiCo')
visual_process_dual_pop_fix(res.history, res.algorithm.problem, problem_name, dir_save)
# visualize_process_one_pop_fix(res.history, res.algorithm.problem, problem_name+'-cscmo-boundary', "G:/code/MyProject/CSCMO/visual_result")
