from surrogate.models.kriging import Kriging
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
from surrogate_problem.surrogate_problem import SurrogateProblemUCBEI

proto = Kriging

pro = MW1()
pop = LatinHypercubeSampling().do(pro, 150)
Evaluator().eval(pro, pop)
surr_pro = SurrogateProblemUCBEI(pro)
surr_pro.fit(pop)
print('--finish fit----')
pop_a = LatinHypercubeSampling().do(pro, 200)
F_ei = surr_pro.evaluate(pop_a.get('X'), ei_flag=True, return_as_dictionary=True)
F_ucb = surr_pro.evaluate(pop_a.get('X'), ei_flag=False, return_as_dictionary=True)
print(F_ei)
print(F_ucb)
# X, F = pop.get('X'), pop.get('F')
# model = ModelSelection(proto).do(X, F[:, 1])
# model.fit(X, F[:, 1])
# test_pop = LatinHypercubeSampling().do(pro, 20)
# o = model.predict(test_pop.get('X'), return_values_of=['y', 'sigma'])
# print(o)
