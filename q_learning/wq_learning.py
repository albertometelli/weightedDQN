import numpy as np
from copy import deepcopy
from mushroom.algorithms.value import TD
from mushroom.utils.table import EnsembleTable
from scipy.stats import norm
import sys
class Gaussian(TD):
    def __init__(self, policy, mdp_info, learning_rate, sigma_learning_rate=None, sigma_1_learning_rate=None, update_mode='deterministic',
                 update_type='weighted', init_values=[0., 0., 500.], delta=0.1, q_max=None, minimize_wasserstein=True):
        self._update_mode = update_mode
        self._update_type = update_type
        self.delta = delta

        self.n_approximators = len(init_values)
        self.Q = EnsembleTable(len(init_values), mdp_info.size)
        if q_max is None:
            q_max = 1 / (1-mdp_info.gamma)
        if self.n_approximators == 3:
            q_max = init_values[0]
            self.sigma_b = init_values[-1]
            self.q_max = q_max

        for i in range(len(self.Q.model)):
            self.Q.model[i].table = np.tile([init_values[i]], self.Q[i].shape)

        super(Gaussian, self).__init__(self.Q, policy, mdp_info,
                                       learning_rate)
        if sigma_learning_rate is None:
            sigma_learning_rate = deepcopy(learning_rate)
        if sigma_1_learning_rate is None:
            sigma_1_learning_rate = deepcopy(learning_rate)
        self.alpha = [deepcopy(self.alpha), deepcopy(sigma_1_learning_rate), deepcopy(sigma_learning_rate)]
        self.minimize_wasserstein = minimize_wasserstein
        policy = np.zeros(self.mdp_info.size)
        self.standard_bound = norm.ppf(1 - self.delta, loc=0, scale=1)
        for s in range(self.mdp_info.size[0]):
            if self.n_approximators == 3:
                means, sigmas1, sigmas2 = [x[[s]] for x in self.Q.model]
                sigmas = sigmas1 + sigmas2
            else:
                means, sigmas = [x[[s]] for x in self.Q.model]
            bounds = sigmas * self.standard_bound + means
            bounds = np.clip(bounds, None, self.q_max)
            actions = np.argwhere(bounds == np.max(bounds)).ravel()
            n = len(actions)
            for a in actions:
                policy[s, a] = 1. / n
        self.policy_matrix = policy
        self.last_update = (0,0)

    def _update(self, state, action, reward, next_state, absorbing):
        raise NotImplementedError


class GaussianQLearning(Gaussian):

    @staticmethod
    def _compute_prob_max(mean_list, sigma_list):
        n_actions = len(mean_list)
        lower_limit = mean_list - 8 * sigma_list
        upper_limit = mean_list + 8 * sigma_list
        epsilon = 1e-5
        _epsilon = 1e-25
        n_trapz = 100
        x = np.zeros(shape=(n_trapz, n_actions))
        y = np.zeros(shape=(n_trapz, n_actions))
        integrals = np.zeros(n_actions)
        for j in range(n_actions):
            if sigma_list[j] < epsilon:
                p = 1
                for k in range(n_actions):
                    if k != j:
                        p *= norm.cdf(mean_list[j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
                integrals[j] = p
            else:
                x[:, j] = np.linspace(lower_limit[j], upper_limit[j], n_trapz)
                y[:, j] = norm.pdf(x[:, j],loc=mean_list[j], scale=sigma_list[j] + _epsilon)
                for k in range(n_actions):
                    if k != j:
                        y[:, j] *= norm.cdf(x[:, j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
                integrals[j] = (upper_limit[j] - lower_limit[j]) / (2 * (n_trapz - 1)) * (y[0, j] + y[-1, j] + 2 * np.sum(y[1:-1, j]))
        #print(np.sum(integrals))
        #assert np.isclose(np.sum(integrals), 1)
        with np.errstate(divide='raise'):
            try:
                return integrals / (np.sum(integrals))
            except FloatingPointError:
                print(integrals)
                print(mean_list)
                print(sigma_list)
                input()


    def _update(self, state, action, reward, next_state, absorbing):
        if self.n_approximators == 3:
            # theoretical version
            mean, sigma1, sigma2 = [x[state, action] for x in self.Q.model]
            if absorbing:
                self.Q.model[0][state, action] = mean + self.alpha[0](state, action) * (
                        reward - mean)
                self.Q.model[1][state, action] = (1 - self.alpha[1](state, action)) * sigma1
                self.Q.model[2][state, action] = self.Q.model[1][state, action] + self.alpha[2](state, action) * self.sigma_b
            else:
                mean_next_all, sigma_next_all1, sigma_next_all2 = \
                    [x[next_state] for x in self.Q.model]
                if self._update_type == 'optimistic':
                    bounds = sigma_next_all2 * self.standard_bound + mean_next_all
                    bounds = np.clip(bounds, -self.q_max, self.q_max)
                    best = np.random.choice(np.argwhere(bounds == np.max(bounds)).ravel())
                    mean_next = mean_next_all[best]
                    sigma_next = sigma_next_all2[best]
                else:
                    raise ValueError("Run pac-GWQL with optimistic estimator")
                self.Q.model[0][state, action] = mean + self.alpha[0](state, action) * (
                        reward + self.mdp_info.gamma * mean_next - mean)
                self.Q.model[1][state, action] = sigma1 + self.alpha[1](state, action) * (
                        self.mdp_info.gamma * sigma_next - sigma1)
                self.Q.model[2][state, action] = self.Q.model[1][state, action] + self.alpha[2](state, action) * self.sigma_b

            mean, sigma1, sigma2 = [x[state, action] for x in self.Q.model]
            bounds = sigma2 * self.standard_bound + mean
            bounds = np.clip(bounds, -self.q_max, self.q_max)

            actions = np.argwhere(bounds == np.max(bounds)).ravel()
            n = len(actions)
            for a in range(self.mdp_info.size[-1]):
                if a in actions:
                    self.policy_matrix[state, a] = 1. / n
                else:
                    self.policy_matrix[state, a] = 0

        else:
            mean, sigma = [x[state, action] for x in self.Q.model]
            sigma = sigma
            self.last_update = (state, action)
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
                        prob = GaussianQLearning._compute_prob_max(mean_next_all, sigma_next_all)
                        mean_next = np.sum(mean_next_all * prob)
                        if self.minimize_wasserstein:
                            sigma_next = np.sum(prob * sigma_next_all)
                        else:
                            sigma_next = np.sum((sigma_next_all + (mean_next - mean_next_all) ** 2) * prob)

                    elif self._update_type == 'optimistic':
                        bounds = sigma_next_all * self.standard_bound + mean_next_all
                        bounds = np.clip(bounds, -self.q_max, self.q_max)
                        best = np.random.choice(np.argwhere(bounds == np.max(bounds)).ravel())
                        mean_next = mean_next_all[best]
                        sigma_next = sigma_next_all[best]

                    else:
                        raise ValueError()

                    self.Q.model[0][state, action] = mean + self.alpha[0](state, action) * (
                            reward + self.mdp_info.gamma * mean_next - mean)
                    self.Q.model[1][state, action] = sigma + self.alpha[1](state, action) * (
                            self.mdp_info.gamma * sigma_next - sigma)

                else:
                    raise NotImplementedError()

    def get_policy(self):
        '''policy = np.zeros(self.mdp_info.size)

        for s in range(self.mdp_info.size[0]):
            bounds = np.zeros(self.mdp_info.size[-1])
            means, sigmas = [x[[s]] for x in self.Q.model]
            for a in range(self.mdp_info.size[-1]):
                bounds[a] = means[a] + norm.ppf(1 - self.delta, loc=means[a],
                                                        scale=sigmas[a] + 1e-15)
            actions = np.argwhere(bounds == np.max(bounds)).ravel()
            n = len(actions)
            for a in actions:
                policy[s, a] = 1. / n'''
        return self.policy_matrix



class GaussianDoubleQLearning(Gaussian):
    def __init__(self, policy, mdp_info, learning_rate, sigma_learning_rate=None, update_mode='deterministic',
                 update_type='weighted', init_values=(0., 500.), delta=0.1, minimize_wasserstein = True):
        super(GaussianDoubleQLearning, self).__init__(
            policy, mdp_info, learning_rate, sigma_learning_rate, update_mode,
            update_type, init_values, delta, minimize_wasserstein)

        self.Qs = [EnsembleTable(2, mdp_info.size),
                   EnsembleTable(2, mdp_info.size)]

        for i in range(len(self.Qs[0])):
            self.Qs[0][i].table = np.tile([init_values[i]], self.Q[i].shape)

        for i in range(len(self.Qs[1])):
            self.Qs[1][i].table = self.Qs[0][i].table.copy()
            self.Q[i].table = self.Qs[0][i].table.copy()
        self.alpha = [deepcopy(self.alpha), deepcopy(self.alpha)]


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
                    prob = GaussianQLearning._compute_prob_max(mean_next_all, sigma_next_all)
                    mean_next = np.sum(mean_next_all_2 * prob)
                    if self.minimize_wasserstein:
                        sigma_next = np.sum(sigma_next_all_2 * prob)
                    else:
                        sigma_next = np.sum((sigma_next_all_2 + (mean_next - mean_next_all_2) ** 2) * prob)
                elif self._update_type == 'optimistic':
                    bounds = np.zeros(self.mdp_info.size[-1])
                    for a in range(self.mdp_info.size[-1]):
                        bounds[a] = mean_next_all[a] + norm.ppf(1 - self.delta, loc=mean_next_all[a], scale=sigma_next_all[a] + 1e-15)
                    best = np.random.choice(np.argwhere(bounds == np.max(bounds)).ravel())
                    mean_next = mean_next_all_2[best]
                    sigma_next = sigma_next_all_2[best]
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
