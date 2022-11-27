from pymoo.core.callback import Callback


class MyCallBack(Callback):

    def __init__(self):

        super(MyCallBack, self).__init__()

    def initialize(self, algorithm):
        self.data['pop_o'] = []
        self.data['pop_h'] = []
        self.data['pop_opt'] = []
        self.data['n_eval'] = []

    def notify(self, algorithm):
        self.data['pop_o'].append(algorithm.pop.get('F'))
        self.data['pop_h'].append(algorithm.pop_h.get('F'))
        self.data['pop_opt'].append(algorithm.opt.get('F'))
        self.data['n_eval'].append(algorithm.evaluator.n_eval)


