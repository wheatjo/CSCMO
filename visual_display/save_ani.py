import pickle
from utils.visualization import display_result, visual_process_dual_pop_fix
import os
from pymoo.visualization.scatter import Scatter
from pymoo.util.optimum import filter_optimum
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

problem_name = 'mw5'
f = open(os.path.join(dir_mytest, 'test', 'pickle_file', 'cscmo_' + problem_name +'_data.pickle'), 'rb')
res = pickle.load(f)
archive_opt = filter_optimum(res.algorithm.archive_o)


dir_save = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result')
visual_process_dual_pop_fix(res.history, res.algorithm.problem, problem_name+'-cscmo-surr', dir_save)

