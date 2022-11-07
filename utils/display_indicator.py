from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.display.column import Column
import numpy as np


class MyOutput(MultiObjectiveOutput):

    def __init__(self):
        super().__init__()
        self.epsilon_k = Column("epsilon_k", width=13)
        self.search_stage = Column("search stage", width=13)
        self.columns += [self.epsilon_k, self.search_stage]

    def update(self, algorithm):
        super().update(algorithm)
        self.epsilon_k.set(algorithm.epsilon_k)
        if algorithm.push_stage:
            self.search_stage.set('push')
        else:
            self.search_stage.set('pull')
            