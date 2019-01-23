from copy import deepcopy
import numpy as np
from mushroom.utils.table import EnsembleTable

class CollectQs:
    """
    This callback can be used to collect the action values in all states at the
    current time step.

    """
    def __init__(self, approximator):
        """
        Constructor.

        Args:
            approximator ([Table, EnsembleTable]): the approximator to use to
                predict the action values.

        """
        self._approximator = approximator

        self._qs = list()

    def __call__(self, **kwargs):
        """
        Add action values to the action-values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        if isinstance(self._approximator, EnsembleTable):
            qs = list()
            for m in self._approximator.model:
                qs.append(m.table)
            self._qs.append(deepcopy(qs))
        else:
            self._qs.append(deepcopy(self._approximator.table))

    def get_values(self):
        """
        Returns:
             The current action-values list.

        """
        return self._qs

class CollectVs:
    """
    This callback can be used to collect the regret

    """
    def __init__(self, mdp, agent, evaluate_policy,frequency =10000):
        """
        Constructor.

        Args:


        """
        self.evaluate_policy = evaluate_policy
        self.mdp = mdp
        self.agent = agent
        self.frequency = frequency
        self.collect = True
        self.count = 0
        self._vs = list()

    def on(self):
        self.collect = True

    def off(self):
        self.collect = False

    def __call__(self, dataset):
        """
        Add action values to the action-values list.

        Args:
            **kwargs (dict): empty dictionary.

        """

        if self.count % self.frequency == 0 and self.collect:
            v_func = list(self.evaluate_policy(self.mdp.p, self.mdp.r, self.agent.get_policy()))

            state = dataset[0][0][0]
            self._vs.append(np.array(v_func+[state]))
            self.count = 0
        if self.collect:
            self.count += 1
    def get_values(self):
        """
        Returns:
             The current action-values list.

        """
        return self._vs
