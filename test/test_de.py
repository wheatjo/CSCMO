from pymoo.algorithms.soo.nonconvex.de import DE, Variant
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi.mw import MW13
from pymoo.optimize import minimize

class NSGA2DE(NSGA2):

    def __init__(self, pop_size=300):
        variant = "DE/rand/1/bin"
        _, selection, n_diffs, crossover = variant.split("/")
        de_operator = Variant(selection=selection, n_diffs=int(n_diffs), crossover=crossover)
        super(NSGA2DE, self).__init__(pop_size=pop_size, mating=de_operator)


alg = NSGA2DE(pop_size=300)
problem = MW13()
res = minimize(problem, alg, ("n_gen", 300), verbose=True)
print(res)






