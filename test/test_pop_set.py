from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.problems.multi.mw import MW3
import numpy as np

p = LatinHypercubeSampling().do(MW3(), 10)
p[np.array([0, 1])].set('a', 1)
o = p[p.get('a') == 1]
print(o)
