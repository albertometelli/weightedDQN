import numpy as np
from copy import deepcopy
from mushroom.algorithms.value import TD
from mushroom.utils.table import EnsembleTable
from scipy.stats import norm

class Particle(TD):
    def __init__(self, policy, mdp_info, learning_rate, sigma_learning_rate=None, update_mode='deterministic',
                 update_type='weighted', init_values=(0., 500.), minimize_wasserstein=True):
        self._update_mode = update_mode
        self._update_type = update_type
        self.Q = EnsembleTable(2, mdp_info.size)

        for i in range(len(self.Q.model)):
            self.Q.model[i].table = np.tile([init_values[i]], self.Q[i].shape)

        super(Particle, self).__init__(self.Q, policy, mdp_info,
                                       learning_rate)
        if sigma_learning_rate is None:
            sigma_learning_rate = deepcopy(learning_rate)
        self.alpha = [deepcopy(self.alpha), deepcopy(sigma_learning_rate)]
        self.minimize_wasserstein = minimize_wasserstein
    def _update(self, state, action, reward, next_state, absorbing):
        raise NotImplementedError


class ParticleQLearning(Particle):

    @staticmethod
    def _compute_prob_max(mean_list, sigma_list):
        n_actions = len(mean_list)
        lower_limit = mean_list - 8 * sigma_list
        upper_limit = mean_list + 8 * sigma_list
        epsilon = 1e2
        n_trapz = 100
        x = np.zeros(shape=(n_trapz, n_actions))
        y = np.zeros(shape=(n_trapz, n_actions))
        integrals = np.zeros(n_actions)
        for j in range(n_actions):
            if sigma_list[j] < epsilon:
                p = 1
                for k in range(n_actions):
                    if k != j:
                        p *= norm.cdf(mean_list[j], loc=mean_list[k], scale=sigma_list[k])
                integrals[j] = p
            else:
                x[:, j] = np.linspace(lower_limit[j], upper_limit[j], n_trapz)
                y[:, j] = norm.pdf(x[:, j],loc=mean_list[j], scale=sigma_list[j])
                for k in range(n_actions):
                    if k != j:
                        y[:, j] *= norm.cdf(x[:, j], loc=mean_list[k], scale=sigma_list[k])
                integrals[j] = (upper_limit[j] - lower_limit[j]) / (2 * (n_trapz - 1)) * (y[0, j] + y[-1, j] + 2 * np.sum(y[1:-1, j]))

        #print(np.sum(integrals))
        #assert np.isclose(np.sum(integrals), 1)
        with np.errstate(divide='raise'):
            try:
                return integrals / np.sum(integrals)
            except FloatingPointError:
                print(integrals)
                print(mean_list)
                print(sigma_list)
                input()


    def _update(self, state, action, reward, next_state, absorbing):
        mean, sigma = [x[state, action] for x in self.Q.model]

        if absorbing:
            self.Q.model[0][state, action] = mean + self.alpha[0](state, action) * (
                        reward - mean)
            self.Q.model[1][state, action] = (1 - self.alpha[1](state, action)) * sigma
        else:
            mean_next_all, sigma_next_all = [x[next_state] for x in self.Q.model]
            if self._update_mode == 'deterministic':
                if self._update_type == 'mean':
                    best = np.random.choice(np.argwhere(mean_next_all == np.max(mean_next_all)).ravel())
                    mean_next = mean_next_all[best]
                    sigma_next = sigma_next_all[best]

                elif self._update_type == 'weighted':
                    prob = ParticleQLearning._compute_prob_max(mean_next_all, sigma_next_all)
                    mean_next = np.sum(mean_next_all * prob)
                    with np.errstate(over='raise'):
                        try:
                            if self.minimize_wasserstein:
                                sigma_next = np.sum(prob * sigma_next_all)
                            else:
                                sigma_next = np.sum((sigma_next_all + (mean_next - mean_next_all) ** 2) * prob)
                        except FloatingPointError:
                            print(prob)
                            print(sigma_next_all)
                            print(mean_next_all)
                            input()
                else:
                    raise ValueError()
                self.Q.model[0][state, action] = mean + self.alpha[0](state, action) * (
                        reward + self.mdp_info.gamma * mean_next - mean)
                self.Q.model[1][state, action] = sigma + self.alpha[1](state, action) * (
                        self.mdp_info.gamma * sigma_next - sigma)
            else:
                raise NotImplementedError()




class ParticleDoubleQLearning(Particle):
    def __init__(self, policy, mdp_info, learning_rate, sigma_learning_rate=None, update_mode='deterministic',
                 update_type='weighted', init_values=(0., 500.)):
        super(ParticleDoubleQLearning, self).__init__(
            policy, mdp_info, learning_rate, update_mode,
            update_type, init_values)

        self.Qs = [EnsembleTable(2, mdp_info.size),
                   EnsembleTable(2, mdp_info.size)]

        for i in range(len(self.Qs[0])):
            self.Qs[0][i].table = np.tile([init_values[i]], self.Q[i].shape)

        for i in range(len(self.Qs[1])):
            self.Qs[1][i].table = self.Qs[0][i].table.copy()
            self.Q[i].table = self.Qs[0][i].table.copy()
        if sigma_learning_rate is None:
            sigma_learning_rate = deepcopy(learning_rate)
        self.alpha = [deepcopy(self.alpha), deepcopy(sigma_learning_rate)]

    def _update(self, state, action, reward, next_state, absorbing):
        if np.random.uniform() < .5:
            i_q = 0
        else:
            i_q = 1

        mean, sigma = np.array([x[state, action] for x in self.Qs[i_q]])
        if absorbing:
            self.Qs[i_q][0][state, action] = mean + self.alpha[i_q][0](state, action) * (
                        reward - mean)
            self.Qs[i_q][0][state, action] = (1 - self.alpha[i_q][1](state, action)) * sigma
            self._update_Q(state, action)
        else:
            mean_next_all, sigma_next_all = [x[next_state] for x in self.Qs[i_q]]
            mean_next_all_2, sigma_next_all_2 = [x[next_state] for x in self.Qs[1 - i_q]]
            if self._update_mode == 'deterministic':
                if self._update_type == 'mean':
                    best = np.random.choice(np.argwhere(mean_next_all == np.max(mean_next_all)).ravel())
                    mean_next = mean_next_all_2[best]
                    sigma_next = sigma_next_all_2[best]

                elif self._update_type == 'weighted':
                    prob = ParticleQLearning._compute_prob_max(mean_next_all, sigma_next_all)
                    mean_next = np.sum(mean_next_all_2 * prob)
                    if self.minimize_wasserstein:
                        sigma_next = np.sum(sigma_next_all_2 * prob)
                    else:
                        sigma_next = np.sum((sigma_next_all_2 + (mean_next - mean_next_all_2) ** 2) * prob)

                else:
                    raise ValueError()
            else:
                raise NotImplementedError()

            self.Qs[i_q][0][state, action] = mean + self.alpha[i_q][0](state, action) * (
                    reward + self.mdp_info.gamma * mean_next - mean)
            self.Qs[i_q][1][state, action] = sigma + self.alpha[i_q][1](state, action) * (
                     self.mdp_info.gamma * sigma_next - sigma)
            self._update_Q(state, action)
    def _update_Q(self, state, action):
        for idx in range(2):
            self.Q[idx][state, action] = np.mean(
                [q[idx][state, action] for q in self.Qs])
