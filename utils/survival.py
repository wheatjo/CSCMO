import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.normalization import normalize
from scipy.spatial.distance import cdist
from pymoo.core.individual import calc_cv
from pymoo.core.population import Population
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
# from fcmeans import FCM
import math
from pymoo.core.problem import Problem
from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.regr import regr_constant
from pymoo.core.duplicate import DefaultDuplicateElimination
from sklearn.cluster import KMeans
from scipy.stats import norm
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.fcm import fcm


class RankAndCrowdingSurvivalIgnoreConstraint(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


class ArchSurvival(Survival):

    def __init__(self, filter_infeasible=False):
        super(ArchSurvival, self).__init__(filter_infeasible)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        # pop: P_t U A_t U Q_t
        f, g, h = pop.get('F', 'G', 'H')
        cv = calc_cv(f, g)
        f = np.column_stack([f, cv])
        nd_front = self.nds.do(f, only_non_dominated_front=True)
        nd_ind = pop[nd_front]

        archive_pop = nd_ind[~nd_ind.get('feasible').squeeze()]
        if len(archive_pop) > 10:
            delete = self.delete(archive_pop, len(archive_pop)-n_survive)
            archive_pop = archive_pop[~(delete == 1)]
        return archive_pop

    @staticmethod
    def delete(pop_v, num_delete):
        F = pop_v.get('F')
        # F_norm = normalize(F)
        nadir = np.max(F, axis=0)
        ideal = np.min(F, axis=0)
        F_norm = (F - nadir) / (ideal - nadir - 1e-10)
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


class ExpensiveArchSurvival(Survival):

    def __init__(self, filter_infeasible=False):
        super(ExpensiveArchSurvival, self).__init__(filter_infeasible)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get('F')


def find_nd_infeasible(pop: Population):
    f = pop.get('F')
    cv = pop.get('CV')
    F = np.column_stack([f, cv])
    nd_ind_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nd_ind = pop[nd_ind_front]
    nd_ind_infeas = nd_ind[~nd_ind.get('feasible').squeeze()]
    return nd_ind_infeas


class EGOSurvival(Survival):

    def __init__(self, archive_pop: Population, problem: Problem, ref_dirs, l1, l2, filter_infeasible=False, theta=None):
        super(EGOSurvival, self).__init__(filter_infeasible)
        self.nds = NonDominatedSorting()
        self.arch_pop = archive_pop
        self.L1 = l1
        self.L2 = l2
        self.problem = problem
        self.model_theta = theta
        self.model = None
        self.fcm = None
        self.ref_dirs = ref_dirs
        self._setup_model()

    def _setup_model(self):
        pop_x = self.arch_pop.get('X')
        n_cluster = 1 + math.ceil((len(self.arch_pop) - self.L1) / self.L2)
        # self.fcm = FCM(n_cluster=n_cluster)
        # self.fcm.fit(pop_x)
        # initialize
        initial_centers = kmeans_plusplus_initializer(pop_x, n_cluster,
                                                      kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()

        # create instance of Fuzzy C-Means algorithm
        self.fcm_instance = fcm(pop_x, initial_centers)

        # run cluster analysis and obtain results
        self.fcm_instance.process()
        # clusters = fcm_instance.get_clusters()
        cluster_centers = self.fcm_instance.get_centers()
        dis_x_center = cdist(pop_x, cluster_centers)
        index = np.argsort(-1 * dis_x_center, axis=0)
        group = index[:self.L1, :]
        self.model = np.empty((n_cluster, self.problem.n_obj), dtype=object)

        if self.model_theta is None:
            self.model_theta = np.ones((n_cluster, self.problem.n_obj))

        for i in range(n_cluster):
            pop_group = self.arch_pop[group[:, i]]
            pop_group_objs = pop_group.get('F')
            pop_group_x = pop_group.get('X')
            for j in range(self.problem.n_obj):
                dacefit = DACE(regr=regr_constant, corr=corr_gauss,
                               theta=5.0, thetaL=0.00001, thetaU=100)
                dacefit.fit(pop_group_x, pop_group_objs[:, j])
                self.model[i, j] = dacefit

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        pop = DefaultDuplicateElimination().do(pop, self.arch_pop)
        kmeans = KMeans(n_clusters=n_survive, random_state=0).fit(pop.get('X'))
        cluster_labels = kmeans.labels_
        ideal_z = np.min(self.arch_pop.get('F'), axis=0)
        g_min = self.get_g_min()
        sel_ind_index = []
        for i in range(n_survive):
            index = np.argwhere(cluster_labels == i)
            if index.size > 1:
                index = index.squeeze()
            else:
                index = index[0]
            pop_cluster_i = pop[index]
            pop_cluster_i_objs, pop_cluster_i_mse = self.model_predict(pop_cluster_i)

            EI = np.zeros(len(index))

            for j in range(len(index)):
                g_te_s = np.max(np.abs(pop_cluster_i_objs[j] - ideal_z) * self.ref_dirs, axis=1)
                min_index = np.argmin(g_te_s)
                EI[j] = self.EI_cal(pop_cluster_i_objs[j], ideal_z, self.ref_dirs[min_index], pop_cluster_i_mse[j], g_min)

            EI_best_cluster_i = np.argmax(EI)
            sel_ind_index.append(index[EI_best_cluster_i])

        sel_ind = pop[sel_ind_index]
        return sel_ind

    def get_g_min(self):
        arch_f = self.arch_pop.get('F')
        z = np.min(arch_f, axis=0)
        g_min_s = []
        for i in range(len(self.arch_pop)):
            g_te_i = np.max(np.abs(arch_f[i] - z) * self.ref_dirs, axis=1)
            g_min_i = np.min(g_te_i)
            g_min_s.append(g_min_i)

        g_min = np.min(np.array(g_min_s))
        return g_min

    def model_predict(self, pop: Population):
        pop_x = pop.get('X')
        if pop_x.ndim < 2:
            pop_x = pop_x[np.newaxis, :]
            pop = Population.merge(Population.new(), pop)
        D = cdist(np.real(pop_x), np.array(self.fcm_instance.get_centers()))
        index = np.argmin(D, axis=1)
        pop_objs = np.zeros((len(pop), self.problem.n_obj))
        pop_objs_mse = np.zeros((len(pop), self.problem.n_obj))
        for i in range(len(pop)):
            for j in range(self.problem.n_obj):
                temp_x = pop[i].get('X')[None, :]
                temp_model = self.model[index[i], j]
                mu, mse = temp_model.predict(temp_x, return_mse=True)
                pop_objs[i, j] = mu
                pop_objs_mse[i, j] = mse

        return pop_objs, pop_objs_mse

    def EI_cal(self, obj, z, lamda, mse, g_best):
        m = self.problem.n_obj
        u = lamda * (obj - z)
        sigma2 = mse
        lamda0 = lamda[0:2]
        mu0 = u[0:2]
        sig20 = sigma2[:2]
        y, s = self.GP_cal(lamda0, mu0, sig20)

        if m >= 3:
            for i in range(2, m):
                lamda0 = np.array([1, lamda[i]])
                mu0 = np.array([y, u[i]])
                sig20 = np.array([s, sigma2[i]])
                y, s = self.GP_cal(lamda0, mu0, np.abs(sig20))

        EI = (g_best - y) * norm.cdf((g_best - y) / np.sqrt(s)) + np.sqrt(s) * norm.pdf((g_best - y) / np.sqrt(s))
        return EI

    def GP_cal(self, lamda, mu, sig2):
        tao = np.sqrt(np.power(lamda[0], 2) * sig2[0] + np.power(lamda[1], 2) * sig2[1])
        alpha = (mu[0] - mu[1]) / tao
        y = mu[0] * norm.cdf(alpha) + mu[1] * norm.cdf(-1.0 * alpha) + tao * norm.pdf(alpha)
        s = (np.power(mu[0], 2) + np.power(lamda[0], 2) * sig2[0]) * norm.cdf(alpha) + \
            (mu[1]**2 + lamda[1]**2 * sig2[1]) * norm.cdf(alpha) + np.sum(mu) * norm.pdf(alpha) - y**2

        return y, s

