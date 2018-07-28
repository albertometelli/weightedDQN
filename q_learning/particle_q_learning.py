from copy import deepcopy

import numpy as np

from mushroom.algorithms.value import TD
from mushroom.utils.table import EnsembleTable


class Particle(TD):
    def __init__(self, policy, mdp_info, learning_rate, n_approximators=10, update_mode='deterministic',
                 update_type='distributional', q_min=0, q_max=1):
        self._n_approximators = n_approximators
        self._update_mode = update_mode
        self._update_type = update_type
        self.Q = EnsembleTable(self._n_approximators, mdp_info.size)
        init_values = np.linspace(q_min, q_max, n_approximators)
        for i in range(len(self.Q.model)):
            self.Q.model[i].table = np.tile([init_values[i]], self.Q[i].shape)

        super(Particle, self).__init__(self.Q, policy, mdp_info,
                                           learning_rate)

        self.alpha = [deepcopy(self.alpha)] * n_approximators

    def _update(self, state, action, reward, next_state, absorbing):
        raise NotImplementedError


class ParticleQLearning(Particle):

    @staticmethod
    def _compute_prob_max(q_list):
        q_array = np.array(q_list).T
        score = (q_array[:, :, None, None] >= q_array).astype(int)
        prob = score.sum(axis=3).prod(axis=2).sum(axis=1)
        return prob / np.sum(prob)

    @staticmethod
    def _compute_max_distribution(q_list):
        q_array = np.array(q_list).T
        n_actions, n_approximators = q_array.shape
        q_array_flat = np.sort(q_array.ravel())
        cdf = (q_array_flat >= q_array[:, :, None]).sum(axis=1).prod(axis=0) / (n_approximators ** n_actions)
        pdf = np.diff(cdf)
        valid_indexes = np.argwhere(pdf != 0).ravel()
        pdf = np.concatenate(([cdf[0]], pdf[valid_indexes]))
        values = np.append(q_array_flat[valid_indexes], q_array_flat[-1])

        return values, pdf

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = np.array([x[state, action] for x in self.Q.model])
        if absorbing:
            for i in range(self._n_approximators):
                self.Q.model[i][state, action] = q_current[i] + self.alpha[i](state, action) * (
                        reward - q_current[i])
        else:
            q_next_all = np.array([x[next_state] for x in self.Q.model])

            if self._update_mode == 'deterministic':
                if self._update_type == 'mean':
                    q_next_mean = np.mean(q_next_all, axis=0)
                    next_index = np.array([np.random.choice(np.argwhere(q_next_mean == np.max(q_next_mean)).ravel())])
                    q_next = q_next_all[:, next_index]
                elif self._update_type == 'distributional':
                    sorted_q_next, pdf = ParticleQLearning._compute_max_distribution(q_next_all)

                    cum_sum = 0.
                    residual_pdf = 0.
                    start = 0
                    q_next = []
                    for i in range(len(pdf)):
                        cum_sum += pdf[i]

                        if cum_sum >= 1. / self._n_approximators:
                            if start > 0:
                                initial_correction = residual_pdf * sorted_q_next[start-1]
                            else:
                                initial_correction = 0.
                            residual_pdf = cum_sum - 1. / self._n_approximators
                            final_correction = (pdf[i] - residual_pdf) * sorted_q_next[i]
                            q = np.sum(pdf[start:i] * sorted_q_next[start:i]) + initial_correction + final_correction
                            q *= self._n_approximators
                            q_next.append(q)
                            while residual_pdf >= 1. / self._n_approximators:
                                q_next.append(sorted_q_next[i])
                                residual_pdf -= 1. / self._n_approximators
                            cum_sum = residual_pdf
                            start = i + 1

                    if cum_sum > 0:
                        q_next.append(sorted_q_next[-1])

                    q_next = np.array(q_next)
                elif self._update_type == 'weighted':
                    prob = ParticleQLearning._compute_prob_max(q_next_all)
                    q_next = np.sum(q_next_all * prob, axis=1)
                else:
                    raise ValueError()
            else:
                raise NotImplementedError()

            for i in range(self._n_approximators):
                self.Q.model[i][state, action] = q_current[i] + self.alpha[i](state, action) * (
                    reward + self.mdp_info.gamma * q_next[i] - q_current[i])

