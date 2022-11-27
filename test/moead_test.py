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

pro = MW1()
pop = LatinHypercubeSampling().do(pro, 190)
Evaluator().eval(pro, pop)
dacefit = DACE(regr=regr_constant, corr=corr_gauss,
               theta=5.0, thetaL=0.00001, thetaU=100)
survival = EGOSurvival(pop, pro, default_ref_dirs(pro.n_obj), l1=80, l2=20)
# dacefit.fit(pop.get('X'), pop.get('F')[:, 0])
pop_test = LatinHypercubeSampling().do(pro, 190)
# i, j = dacefit.predict(pop_test.get('X'), return_mse=True)
pop_test_f = pro.evaluate(pop_test.get('X'), return_values_of=['F'])
pop_sur = survival.do(pro, pop_test, n_survive=5)
print(pop_sur)
Evaluator().eval(pro, pop_sur)

plt.plot(pop_test_f[:, 0], pop_test_f[:, 1], 'go', markerfacecolor='none', label='test')
plt.plot(pop_sur.get('F')[:, 0], pop_sur.get('F')[:, 1], 'r.', label='surr')
plt.scatter(pop.get('F')[:, 0], pop.get('F')[:, 1], label='data')

plt.legend()
plt.show()

