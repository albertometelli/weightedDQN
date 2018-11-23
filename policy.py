import numpy as np

from mushroom.policy.td_policy import TDPolicy
from mushroom.utils.parameters import Parameter
from scipy.stats import norm


class BootPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(BootPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False
        self._idx = None

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                max_as, count = np.unique(np.argmax(q_list, axis=1),
                                          return_counts=True)
                max_a = np.array([max_as[np.random.choice(
                    np.argwhere(count == np.max(count)).ravel())]])

                return max_a
            else:
                q = self._approximator.predict(state, idx=self._idx)
                
                max_a = np.argwhere(q == np.max(q)).ravel()
                if len(max_a) > 1:
                    max_a = np.array([np.random.choice(max_a)])

                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        self._idx = idx

    def update_epsilon(self, state):
        self._epsilon(state)


class WeightedPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(WeightedPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False

    @staticmethod
    def _compute_prob_max(q_list):
        q_array = np.array(q_list).T
        score = (q_array[:, :, None, None] >= q_array).astype(int)
        prob = score.sum(axis=3).prod(axis=2).sum(axis=1)
        prob = prob.astype(np.float32)
        return prob / np.sum(prob)

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                prob = WeightedPolicy._compute_prob_max(q_list)
                max_a = np.array([np.random.choice(np.argwhere(prob == np.max(prob)).ravel())])
                return max_a
            else:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for i in range(self._n_approximators):
                        q_list.append(self._approximator.predict(state, idx=i))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                qs = np.array(q_list)

                samples = np.ones(self._approximator.n_actions)
                for a in range(self._approximator.n_actions):
                    idx = np.random.randint(self._n_approximators)
                    samples[a] = qs[idx, a]

                max_a = np.array([np.random.choice(np.argwhere(samples == np.max(samples)).ravel())])
                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])
            
    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass

    def update_epsilon(self, state):
        self._epsilon(state)

class VPIPolicy(TDPolicy):

    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(VPIPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False


    @staticmethod
    def _count_with_ties(q, mu, sign):
        if sign == '<':
            disqual = np.sum(q < mu)
        elif sign == '>':
            disqual = np.sum(q > mu)
        else:
            raise ValueError()
        equal = np.sum(q == mu)
        return disqual + equal / 2.


    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                mean_q = np.mean(q_list, axis=0)
                max_a = np.array([np.random.choice(np.argwhere(mean_q == np.max(mean_q)).ravel())])
                return max_a
            else:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for i in range(self._n_approximators):
                        q_list.append(self._approximator.predict(state, idx=i))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                qs = np.array(q_list)

                mean_q = np.mean(q_list, axis=0)
                best2_idx = np.argpartition(-mean_q, 1)[:2]
                a1, a2 = best2_idx if mean_q[best2_idx[0]] >= mean_q[best2_idx[1]] else np.flip(best2_idx)
                mu1, mu2 = mean_q[a1], mean_q[a2]

                assert mu1 >= mu2
                assert mu1 >= max(mean_q)

                vpi = np.zeros(self._approximator.n_actions)
                for a in range(self._approximator.n_actions):
                    if a == a1:
                        count = VPIPolicy._count_with_ties(qs[:, a], mu2, '<')
                        if count == 0:
                            vpi[a] = 0
                        else:
                            vpi[a] = 1. / count * np.sum(np.clip(mu2 - qs[:, a], 0, np.inf))
                    else:
                        count = VPIPolicy._count_with_ties(qs[:, a], mu1, '>')
                        if count == 0:
                            vpi[a] = 0
                        else:
                            vpi[a] = 1. / count * np.sum(np.clip(qs[:, a] - mu1, 0, np.inf))

                score = mean_q + vpi

                max_a = np.array([np.random.choice(np.argwhere(score == np.max(score)).ravel())])

                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass

    def update_epsilon(self, state):
        self._epsilon(state)


class WeightedGaussianPolicy(TDPolicy):
    def __init__(self, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(WeightedGaussianPolicy, self).__init__()

        self._epsilon = epsilon
        self._evaluation = False

    @staticmethod
    def _compute_prob_max(q_list):
        mean_list = q_list[0]
        sigma_list = q_list[1]
        n_actions = len(mean_list)
        lower_limit = mean_list - 8 * sigma_list
        upper_limit = mean_list + 8 * sigma_list
        n_trapz = 100
        x = np.zeros(shape=(n_trapz, n_actions))
        y = np.zeros(shape=(n_trapz, n_actions))
        for j in range(n_actions):
            x[:, j] = np.linspace(lower_limit[j], upper_limit[j], n_trapz)
            y[:, j] = norm.pdf(x[:, j], loc=mean_list[j], scale=sigma_list[j])
            for k in range(n_actions):
                if k != j:
                    y[:, j] *= norm.cdf(x[:, j], loc=mean_list[k], scale=sigma_list[k])

        integrals = ((upper_limit - lower_limit) / (2 * (n_trapz - 1))) * \
                    (y[0, :] + y[-1, :] + 2 * np.sum(y[1:-1, :], axis=0))
        #print(np.sum(integrals))
        #assert np.isclose(np.sum(integrals), 1)
        return integrals

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                qs = np.array(q_list)
                means = qs[0, :]
                max_a = np.array([np.random.choice(np.argwhere(means == np.max(means)).ravel())])
                return max_a
            else:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                qs = np.array(q_list)

                samples = np.ones(self._approximator.n_actions)
                for a in range(self._approximator.n_actions):
                    mean = qs[0, a]
                    sigma = qs[1, a]
                    samples[a] = np.random.normal(loc=mean, scale=sigma)

                max_a = np.array([np.random.choice(np.argwhere(samples == np.max(samples)).ravel())])
                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass

    def update_epsilon(self, state):
        self._epsilon(state)