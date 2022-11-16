from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, binary_tournament
from pymoo.core.mating import Mating
from pymoo.operators.selection.tournament import compare, TournamentSelection
from utils.survival import RankAndCrowdingSurvivalIgnoreConstraint
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import copy
from pymoo.algorithms.soo.nonconvex.de import Variant
from pymoo.operators.control import NoParameterControl


class CCMO(GeneticAlgorithm):
    # problem is surrogate model
    def __init__(self, pop_o_init, pop_size, n_offspring, **kwargs):
        super(CCMO, self).__init__(pop_size=pop_size, sampling=pop_o_init, n_offspring=n_offspring,
                                   output=MultiObjectiveOutput(), advance_after_initial_infill=True,**kwargs)

        self.pop_h = None
        self.survival = RankAndCrowdingSurvival()
        self.survival_help = RankAndCrowdingSurvivalIgnoreConstraint()
        self.mating_o = Mating(selection=TournamentSelection(func_comp=binary_tournament),
                               crossover=SBX(eta=15, prob=0.9),
                               mutation=PM(eta=20),
                               repair=self.repair,
                               eliminate_duplicates=self.eliminate_duplicates,
                               n_max_iterations=100)
        # mating_h
        self.mating_h = Variant(selection='rand', n_diffs=1, crossover='bin', control=NoParameterControl)
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _infill(self):
        off_o = self.mating_o.do(problem=self.problem, pop=self.pop, n_offsprings=self.n_offsprings, algorithm=self)
        off_h = self.mating_o.do(problem=self.problem, pop=self.pop_h, n_offsprings=self.n_offsprings, algorithm=self)
        self.pop = Population.merge(self.pop, off_o, off_h)
        self.pop_h = Population.merge(self.pop_h, off_o, off_h)
        self.evaluator.eval(self.problem, self.pop)
        self.evaluator.eval(self.problem, self.pop_h)

        self.pop = self.survival.do(problem=self.problem, pop=self.pop, n_survive=self.pop_size)
        self.pop_h = self.survival_help.do(problem=self.problem, pop=self.pop_h, n_survive=self.pop_size)

    def _initialize_advance(self, infills=None, **kwargs):
        # self.pop_h = Population.new(X=self.pop.get('X'), F=self.pop.get('F'))
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills)
        self.pop_h = copy.deepcopy(self.pop)
    # def _setup(self, problem, **kwargs):
