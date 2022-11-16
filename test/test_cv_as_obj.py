from pymoo.constraints.as_obj import ConstraintsAsObjective
from DisplayProblem.displaymw import *
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.core.survival import Survival
from pymoo.core.individual import calc_cv
from pymoo.util.normalization import normalize
from scipy.spatial.distance import cdist


class CvAsObjectSurvival(Survival):

    def __init__(self, filter_infeasible=False):
        super(CvAsObjectSurvival, self).__init__(filter_infeasible)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        # pop: P_t U A_t U Q_t
        f, g, h = pop.get('F', 'G', 'H')
        cv = calc_cv(f, g)
        f = np.column_stack([f, cv])
        nd_front = self.nds.do(f, only_non_dominated_front=True)
        nd_ind = pop[nd_front]

        archive_pop = nd_ind[~nd_ind.get('feasible').squeeze()]
        delete = self.delete(archive_pop, len(archive_pop)-n_survive)
        archive_pop = archive_pop[~(delete == 1)]
        return archive_pop

    @staticmethod
    def delete(pop_v, num_delete):
        F = pop_v.get('F')
        F_norm = normalize(F)
        cosine = 1 - cdist(F_norm, F_norm, 'cosine')
        cosine = cosine * (1 - np.eye(len(F)))
        delete_index = np.zeros(len(pop_v))
        while np.sum(delete_index) < num_delete:
            ind_min_angle_index = np.argwhere(cosine == np.max(cosine))[0]
            ind1 = pop_v[ind_min_angle_index[0]]
            ind2 = pop_v[ind_min_angle_index[1]]
            if ind1.get('CV') < ind2.get('CV') or (ind1.get('CV') == ind2.get('CV') and np.random.random() < 0.5):
                delete_index[ind_min_angle_index[1]] = 1
                cosine[:, ind_min_angle_index[1]] = 0
                cosine[ind_min_angle_index[1], :] = 0

            else:
                delete_index[ind_min_angle_index[0]] = 1
                cosine[:, ind_min_angle_index[0]] = 0
                cosine[ind_min_angle_index[0], :] = 0

        return delete_index



if __name__ == '__main__':
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling
    import pickle
    import os
    from utils.visualization import display_result
    problem_name = 'mw3'
    dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = open(os.path.join(dir_mytest, 'test', 'pickle_file', 'cscmo_' + problem_name + '_data.pickle'), 'rb')
    res = pickle.load(f)
    pop_o, pop_h = res.algorithm.pop, res.algorithm.pop_h
    problem = res.algorithm.problem
    archive = CvAsObjectSurvival().do(problem, Population.merge(pop_o, pop_h), n_survive=20)
    print(len(archive))
    display_result(problem, Population.merge(pop_o, pop_h).get('F'))
    display_result(problem, archive.get('F'))
