import sys
import os

dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_mytest)

from utils.SelectCand import select_exploit_explore_ind_simple
from pymoo.core.individual import Individual
import numpy as np
from pymoo.core.population import Population
from pymoo.problems.multi.mw import MW3
from pymoo.core.evaluator import Evaluator
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

from pymoo.algorithms.moo.nsga2 import binary_tournament, NSGA2
from pymoo.core.mating import Mating
from pymoo.operators.selection.tournament import TournamentSelection

from pymoo.operators.mutation.pm import PM

from pymoo.core.duplicate import DefaultDuplicateElimination

from pymoo.core.repair import NoRepair
from pymoo.operators.crossover.sbx import SBX
from utils.survival import RankAndCrowdingSurvivalIgnoreConstraint
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import os

mating_o = Mating(selection=TournamentSelection(func_comp=binary_tournament),
                  crossover=SBX(eta=15, prob=0.9),
                  mutation=PM(eta=20), repair=NoRepair(),
                  eliminate_duplicates=DefaultDuplicateElimination(),
                  n_max_iterations=100)
mating = NSGA2().mating
survival = RankAndCrowdingSurvivalIgnoreConstraint()
evaluator = Evaluator()
problem = MW3()
pop_init = LatinHypercubeSampling().do(problem, 200)
evaluator.eval(problem, pop_init)

archive = LatinHypercubeSampling().do(problem, 50)
evaluator.eval(problem, archive)
pop_init = survival.do(problem, pop_init)
I = NonDominatedSorting().do(pop_init.get('F'), only_non_dominated_front=True)
opt = pop_init[I]

pop_sel = select_exploit_explore_ind_simple(pop_init, archive, n_exploit=5, n_explore=5, mating=mating,
                                            problem=problem, help_flag=False, alg=NSGA2())

print(pop_sel)
