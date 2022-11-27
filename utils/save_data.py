from utils.visualization import visual_dual_pop
from pymoo.core.result import Result
from pymoo.util.display.output import pareto_front_if_possible
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.optimum import filter_optimum
import os
import time
import yaml
import pickle


class SaveData(object):

    def __init__(self, alg_name: str, problem_name: str, max_eval: int, res: Result, opt_problem, benchmark_flag=False,
                 runner_num=None):
        super(SaveData, self).__init__()
        self.algorithm = res.algorithm
        self.res = res
        self.opt_problem = self.algorithm.problem
        self.problem_name = problem_name
        self.algorithm_name = alg_name
        self.message = {'algorithm': self.algorithm_name, 'opt_problem': problem_name, 'max_eval': max_eval}
        self.indicator = ['igd', 'hv', 'igd+']
        self.save_path = None
        self.benchmark_flag =benchmark_flag
        self.opt_problem = opt_problem
        if runner_num is not None:
            self.runner_num = runner_num
            self.message['runner_num'] = runner_num
        else:
            self.runner_num = runner_num

    def process(self, save_ani=False, save_archive=False):
        self.initialize()
        # pop is objs
        print(yaml.dump(self.message, sort_keys=False))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(os.path.join(self.save_path, 'message.yml'), 'w') as f:
            yaml.dump(self.message, f, sort_keys=False)

        if save_archive:
            with open(os.path.join(self.save_path, 'archive.pickle'), 'wb') as f:
                pickle.dump(self.res.algorithm.archive_all, f)

        if save_ani:
            rec_callback = self.algorithm.callback.data
            pop_o_history = rec_callback['pop_o']
            pop_h_history = rec_callback['pop_h']
            pop_opt_history = rec_callback['pop_opt']
            pf = self.opt_problem.pareto_front()
            n_eval = rec_callback['n_eval']
            pop_history = {'pop_o': pop_o_history, 'pop_h': pop_h_history,
                           'pop_opt': pop_opt_history, 'pf': pf, 'n_eval': n_eval}
            if self.opt_problem.n_obj > 2:
                return
            visual_dual_pop(pop_history, self.opt_problem, self.problem_name,
                            os.path.join(self.save_path, self.problem_name+'.gif'))

    def get_indicator(self):
        archive_pop = self.algorithm.archive_all
        archive_pop_opt = filter_optimum(archive_pop)
        pf = pareto_front_if_possible(self.algorithm.problem)
        if archive_pop_opt is not None:
            objs = archive_pop_opt.get('F')
            self.message['igd'] = float(IGD(pf, zero_to_one=True)(objs))
            self.message['igd+'] = float(IGDPlus(pf, zero_to_one=True)(objs))
            self.message['hv'] = float(Hypervolume(pf=pf, zero_to_one=True)(objs))

    def initialize(self):
        self.message['pop_size'] = self.res.algorithm.pop_size
        self.message['exec_time'] = self.res.exec_time
        self.get_indicator()

        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

        if self.benchmark_flag:
            result_save_path = os.path.join(project_path, 'benchmark_test', self.algorithm_name, self.problem_name,
                                            folder_name+f"-runner_num-{self.runner_num}")

        else:
            result_save_path = os.path.join(project_path, 'result_file', self.algorithm_name,
                                            self.problem_name, folder_name)

        self.save_path = result_save_path


if __name__ == '__main__':

    from pymoo.problems.multi.mw import MW13

    with open('G:/code/MyProject/CSCMO/result_file/mw13/2022-11-24 17-18-35/archive.pickle', 'rb') as f:
        res = pickle.load(f)

    print(res)
    p = res.sols
    p = filter_optimum(p)
    f = p.get('F')
    igd = IGD(pf=MW13().pareto_front(), zero_to_one=True)
    value = igd.do(f)
    print(value)
    p = {'igd': float(value)}
    with open('a.yml', 'w') as f:
        yaml.dump(p, f)

