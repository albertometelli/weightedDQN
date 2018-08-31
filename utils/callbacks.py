from copy import deepcopy

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
