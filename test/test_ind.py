from pymoo.core.individual import Individual, calc_cv
import numpy as np
from pymoo.core.population import Population
from pymoo.problems.multi.mw import MW3
from pymoo.core.evaluator import Evaluator
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
import copy

evaluator = Evaluator()
problem = MW3()
pop_init = LatinHypercubeSampling().do(problem, 200)
evaluator.eval(problem, pop_init)

fes_index = pop_init.get('feasible')
print(fes_index)
CV_mat = pop_init.get('CV')
res = problem.evaluate(pop_init.get('X'), return_as_dictionary=True)
cv_res = calc_cv(res.get('G'), res.get('H'))
print(cv_res[cv_res.argsort()])
cv_off = pop_init[cv_res.argsort()]

print(CV_mat)
CV_mat = CV_mat - 13
pop_init_copy = copy.deepcopy(pop_init)
pop_init_copy.set('CV', CV_mat)
print((pop_init.get('feasible')))
print((pop_init_copy.get('feasible')))

pop_init.set('archive', 'help')
print(pop_init.get('archive') == 'help')
pop_init[0].set('evaluated', set())
print(len(pop_init[0].evaluated))


