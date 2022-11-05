import sys
import os

dir_mytest = os.path.dirname(os.path.abspath(__file__))

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.pcx import PCX
import pickle

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("mw3")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=True, save_history=True)

f = open(os.path.join(dir_mytest, 'pickle_file', 'b.pickle'), 'wb')
pickle.dump(res, f)
# plot = Scatter()
# plot.add(problem.pareto_front(), color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()
