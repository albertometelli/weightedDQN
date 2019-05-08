from mushroom.utils.table import Table
import numpy as np
class Parameter(object):
    """
    This class implements function to manage parameters, such as learning rate.
    It also allows to have a single parameter for each state of state-action
    tuple.
    """
    def __init__(self, value, min_value=None, size=(1,)):
        """
        Constructor.
        Args:
            value (float): initial value of the parameter;
            min_value (float): minimum value that it can reach when decreasing;
            size (tuple, (1,)): shape of the matrix of parameters; this shape
                can be used to have a single parameter for each state or
                state-action tuple.
        """
        self._initial_value = value
        self._min_value = min_value
        self._n_updates = Table(size)

    def __call__(self, *idx, **kwargs):
        """
        Update and return the parameter in the provided index.
        Args:
             *idx (list): index of the parameter to return.
        Returns:
            The updated parameter in the provided index.
        """
        if self._n_updates.table.size == 1:
            idx = list()

        self.update(*idx, **kwargs)

        return self.get_value(*idx, **kwargs)

    def get_value(self, *idx, **kwargs):
        """
        Return the current value of the parameter in the provided index.
        Args:
            *idx (list): index of the parameter to return.
        Returns:
            The current value of the parameter in the provided index.
        """
        new_value = self._compute(*idx, **kwargs)

        if self._min_value is None or new_value >= self._min_value:
            return new_value
        else:
            return self._min_value

    def _compute(self, *idx, **kwargs):
        """
        Returns:
            The value of the parameter in the provided index.
        """
        return self._initial_value

    def update(self, *idx, **kwargs):
        """
        Updates the number of visit of the parameter in the provided index.
        Args:
            *idx (list): index of the parameter whose number of visits has to be
                updated.
        """
        self._n_updates[idx] += 1

    @property
    def shape(self):
        """
        Returns:
            The shape of the table of parameters.
        """
        return self._n_updates.table.shape





class LogarithmicDecayParameter(Parameter):

    def __init__(self, value, C=1., min_value=None, size=(1,)):
        self._C = C #(2R/(c np.sqrt(2 pi) (1-gamma) sigma_0))
        super(LogarithmicDecayParameter, self).__init__(value, min_value, size)

    def _compute(self, *idx, **kwargs):
        n = np.maximum(self._n_updates[idx], 1)
        lr = 1 - np.e ** (-(1 / (n+1) * (self._C + 2 * np.log(n + 1))))
        return lr
