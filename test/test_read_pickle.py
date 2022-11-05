import pickle
from utils.visualization import visualize_process_test, display_result
import os
from pymoo.visualization.scatter import Scatter
from pymoo.util.optimum import filter_optimum
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
dir_mytest = os.path.dirname(os.path.abspath(__file__))

problem_name = 'mw13'
f = open(os.path.join(dir_mytest, 'pickle_file', 'cscmo_' + problem_name +'_data.pickle'), 'rb')
res = pickle.load(f)
archive_opt = filter_optimum(res.algorithm.archive_o)
plot = Scatter()
plot.add(res.algorithm.problem.pareto_front(), color="black", linewidth=2)
plot.add(filter_optimum(res.algorithm.archive_o).get('F'),  color='red', s=30, edgecolors='r')

dir_save = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_result', problem_name + '.png')
plot.save(dir_save, dpi=300)





