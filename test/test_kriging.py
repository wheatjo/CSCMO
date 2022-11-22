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


proto = Kriging

pro = MW1()
pop = LatinHypercubeSampling().do(pro, 160)
Evaluator().eval(pro, pop)
X, F = pop.get('X'), pop.get('F')
model = ModelSelection(proto).do(X, F[:, k])
model.fit(X, F[:, k])