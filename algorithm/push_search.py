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
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.algorithms.moo.moead import NeighborhoodSelection
from pymoo.core.duplicate import NoDuplicateElimination
from pysamoo.core.archive import Archive
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.optimum import filter_optimum

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
        self.archive_all = filter_optimum(infills)

    def _advance(self, infills=None, **kwargs):
        temp = Population.merge(self.pop, self.pop_h)

        self.archive_all = Population.merge(self.archive_all, filter_optimum(temp))

class MOEADEGO(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_neighbors=20,
                 decomposition=None,
                 prob_neighbor_mating=0.9,
                 sampling=LatinHypercubeSampling(),
                 crossover=SBX(prob=1.0, eta=20),
                 mutation=PM(prob_var=None, eta=20),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        self.ref_dirs = ref_dirs

        # the decomposition metric used
        self.decomposition = decomposition

        # the number of neighbors considered during mating
        self.n_neighbors = n_neighbors

        self.neighbors = None

        self.selection = NeighborhoodSelection(prob=prob_neighbor_mating)

        super().__init__(pop_size=len(ref_dirs),
                         sampling=sampling,
                         crossover=crossover,
                         mutation=mutation,
                         eliminate_duplicates=NoDuplicateElimination(),
                         output=output,
                         advance_after_initialization=False,
                         **kwargs)

        def _setup(self, problem, **kwargs):
            assert not problem.has_constraints(), "This implementation of MOEAD does not support any constraints."

            # if no reference directions have been provided get them and override the population size and other settings
            if self.ref_dirs is None:
                self.ref_dirs = default_ref_dirs(problem.n_obj)
            self.pop_size = len(self.ref_dirs)

            # neighbours includes the entry by itself intentionally for the survival method
            self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:,
                             :self.n_neighbors]

            # if the decomposition is not set yet, set the default
            if self.decomposition is None:
                self.decomposition = default_decomp(problem)