from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.pcx import PCX

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("mw9")

algorithm = NSGA2(pop_size=200)

res = minimize(problem,
               algorithm,
               ('n_eval', 79600),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()