from surrogate.models.rbf import RBF
from surrogate.selection import ModelSelection
from DisplayProblem.displaymw import DisplayMW2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.evaluator import Evaluator

problem = DisplayMW2()
pop = LatinHypercubeSampling().do(problem, 100)
Evaluator().eval(problem, pop)
proto = RBF
X, F = pop.get('X'), pop.get('F')
p = LatinHypercubeSampling().do(problem, 100)
model = ModelSelection(proto).do(X, F[:, 1])
model.fit(X, F[:, 1])
prd = model.predict(p.get('X'))
org = problem.evaluate(p.get('X'), return_values_of=['F'])
print(model)
print(prd)
print(org[:, 1])
