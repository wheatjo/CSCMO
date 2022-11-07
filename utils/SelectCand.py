from sklearn.cluster import KMeans
from pymoo.util.roulette import RouletteWheelSelection
from pymoo.core.population import Population
from pymoo.util.normalization import normalize
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.core.mating import Mating
from pymoo.core.problem import Problem
from pymoo.util.misc import norm_eucl_dist
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import calc_cv
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from utils.help_problem import MinProblemCV, FindCvEdge
from pymoo.optimize import minimize
from pymoo.core.evaluator import Evaluator
from utils.survival import RankAndCrowdingSurvivalIgnoreConstraint
from algorithm.multimodal import TabuCV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def cluster_and_choose(opt: Population, n_cluster):

    if len(opt) <= n_cluster:
        return [i for i in range(len(opt))]

    choose_index = []
    norm_f_pop1 = normalize(opt.get('F'))
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(norm_f_pop1)
    groups = [[] for _ in range(n_cluster)]
    for k, i in enumerate(kmeans.labels_):
        groups[i].append(k)

    for group in groups:
        if len(group) > 0:
            fitness = opt[group].get('crowding').argsort()
            selection = RouletteWheelSelection(fitness, larger_is_better=False)
            index = group[selection.next()]
            choose_index.append(index)

    return choose_index


def my_select_points_with_maximum_distance(problem, X, others, n_select, selected=[]):
    n_points, n_dim = X.shape

    # calculate the distance matrix
    D = norm_eucl_dist(problem, X, X)
    dist_to_others = norm_eucl_dist(problem, X, others)
    D = np.column_stack([D, dist_to_others])

    # if no selection provided
    if len(selected) == 0:
        selected = [dist_to_others.min(axis=1).argmax()]

    # create variables to store what selected and what not
    not_selected = [i for i in range(n_points) if i not in selected]

    # now select the points until sufficient ones are found
    while len(selected) < n_select:
        # find point that has the maximum distance to all others
        index_in_not_selected = D[not_selected][:, selected].min(axis=1).argmax()
        I = not_selected[index_in_not_selected]

        # add it to the selected and remove from not selected
        selected.append(I)
        not_selected = [i for i in range(n_points) if i not in selected]

    return selected


def generate_explore_offspring_simple(pop1: Population, pop2: Population, algorithm) -> Population:
    offspring_1 = algorithm.generate_off_mating.do(problem=algorithm.problem, pop=pop1,
                                                   n_offsprings=algorithm.surr_n_offspring)

    algorithm.evaluator.eval(algorithm.surrogate_problem, offspring_1, count_evals=False)
    offspring_1 = offspring_1[offspring_1.get('feasible').squeeze()]
    crowding_distance = calc_crowding_distance(offspring_1.get('F'))
    pop1_explore = offspring_1[crowding_distance.argsort()[-algorithm.n_off_explore:]]

    return pop1_explore


def select_exploit_individuals(pop_o: Population, pop_h: Population, n_exploit, n_explore) -> (Population, Population):

    pop_o_exploit = pop_o[cluster_and_choose(pop_o, n_exploit)]
    pop_h_exploit = pop_h[cluster_and_choose(pop_h, n_exploit)]
    return pop_o_exploit, pop_h_exploit


def select_exploit_explore_ind_simple(pop: Population, archive: Population,
                                      n_exploit, n_explore, mating: Mating, problem: Problem, help_flag, alg) -> tuple:

    if help_flag:
        pop = RankAndCrowdingSurvivalIgnoreConstraint().do(problem, pop)

    pop_exploit = pop[cluster_and_choose(pop, n_exploit)]
    off = mating.do(problem, pop_exploit, n_explore*100, algorithm=alg)

    if help_flag is False:
        res = problem.evaluate(off.get('X'), return_as_dictionary=True)
        cv_mat = calc_cv(res.get('G'), res.get('H'))
        off_fes = off[cv_mat <= 0]
        if off_fes.size < n_exploit:
            choose_num = n_exploit - off_fes.size
            off_infes = off[cv_mat[cv_mat > 0].argsort()][:choose_num]
            off = Population.merge(off_fes, off_infes)

        others = Population.merge(archive, pop_exploit)
        I = my_select_points_with_maximum_distance(problem, off.get('X'), others.get('X'), n_explore)

        pop_explore = off[I]

    else:

        others = Population.merge(archive, pop_exploit)
        I = my_select_points_with_maximum_distance(problem, off.get('X'), others.get('X'), n_explore)
        pop_explore = off[I]

    return pop_exploit, pop_explore


def pull_stage_search(pop: Population, archive: Population,
                      n_exploit, n_explore, mating: Mating, problem: Problem, help_flag, alg) -> tuple:

    cv_obj_problem = MinProblemCV(problem)
    cv_opt_pop = Population.new(X=pop.get('X'))
    multi_modal_cv_alg = NicheGA(pop_size=100, n_offsprings=100, sampling=cv_opt_pop)
    cv_opt_res = minimize(cv_obj_problem, multi_modal_cv_alg, ('n_gen', 100))
    cv_opt_cand = cv_opt_res.pop[cv_opt_res.pop.get('F')[:, 0] == 0]
    cv_opt_cand = Population.new(X=cv_opt_cand.get('X'))
    pop_h = Population.merge(pop, cv_opt_cand)
    Evaluator().eval(problem, pop_h)

    pop_h_exploit, pop_h_explore = select_exploit_explore_ind_simple(pop_h, archive, n_exploit, n_explore, mating,
                                                                     problem, help_flag=help_flag, alg=alg)

    return pop_h_exploit, pop_h_explore


def pull_stage_explore(pop: Population, problem: Problem, n_explore, edge_epsilon):

    cv_edge_problem = FindCvEdge(problem, edge_epsilon)
    cv_edge_opt_pop = Population.new(X=pop.get('X'))
    multi_modal_cv_alg = TabuCV(pop_size=n_explore, n_offsprings=n_explore, sampling=cv_edge_opt_pop)
    res = minimize(cv_edge_problem, multi_modal_cv_alg, ('n_gen', 100), verbose=True)
    res_pop = Population.new(X=res.pop.get('X'))
    Evaluator().eval(problem, res_pop)
    res_pop_feasible = res_pop[res_pop.get('feasible')[:, 0]]
    res_pop_feasible = RankAndCrowdingSurvivalIgnoreConstraint().do(problem, res_pop_feasible)
    fronts = NonDominatedSorting().do(res_pop_feasible.get('F'), n_stop_if_ranked=len(res_pop_feasible))

    if len(res_pop_feasible) < n_explore:
        pop_h_explore = res_pop_feasible
        return pop_h_explore

    else:
        index = np.ones(len(res_pop_feasible)) > 0
        index[fronts[-1]] = False
        res_cand = res_pop_feasible[index]
        cand = res_cand[cluster_and_choose(res_cand, n_explore)]
        return cand


if __name__ == '__main__':
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling
    from pymoo.problems import get_problem
    from visualization import display_result
    from DisplayProblem.displayctp import *
    # problem = get_problem('ctp4', n_var=10)
    problem = DisplayCTP4(n_var=10)
    pop = LatinHypercubeSampling().do(problem, 200)
    sel = pull_stage_explore(pop, problem, 1000, 0.1)

    print(len(sel))
    if len(sel) == 0:
        print('no find')
    else:
        display_result(problem, sel.get('F'))




