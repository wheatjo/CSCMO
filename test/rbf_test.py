from surrogate.models.rbf import RBF
from surrogate.selection import ModelSelection
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.reference_direction import default_ref_dirs
from pymoo.problems.multi.mw import MW1
from pymoo.core.evaluator import Evaluator
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from utils.survival import EGOSurvival
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant
from pydacefit.corr import corr_gauss
import matplotlib.pyplot as plt
from surrogate_problem.surrogate_problem import SurrogateProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

pro = MW1()
pop = LatinHypercubeSampling().do(pro, 160)
Evaluator().eval(pro, pop)
surr_pro = SurrogateProblem(pro)
surr_pro.fit(pop)
pop_test = LatinHypercubeSampling().do(pro, 160)
Evaluator().eval(surr_pro, pop_test)
nds = NonDominatedSorting()
front = NonDominatedSorting().do(pop_test.get('F'), only_non_dominated_front=True)
print(front)
pop_nds = pop_test[front]
pop_sur_f = pro.evaluate(pop_nds.get('X'), return_values_of=['F'])
pop_test_f = pro.evaluate(pop_test.get('X'), return_values_of=['F'])
plt.plot(pop_test_f[:, 0], pop_test_f[:, 1], 'go', markerfacecolor='none', label='test')
plt.plot(pop_sur_f[:, 0], pop_sur_f[:, 1], 'r.', label='surr')
plt.scatter(pop.get('F')[:, 0], pop.get('F')[:, 1], label='data')
plt.legend()
plt.show()
