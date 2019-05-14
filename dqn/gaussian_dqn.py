from copy import deepcopy

import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor
from scipy.stats import norm
from replay_memory import ReplayMemory


class GaussianDQN(Agent):
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 target_update_frequency, initial_replay_size,
                 max_replay_size, fit_params=None, approximator_params=None, clip_reward=True,
                 update_type='weighted', delta=0.1, store_prob=False):
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency
        self.update_type = update_type
        self.delta = delta
        self.standard_bound = norm.ppf(1 - self.delta, loc=0, scale=1)
        self.store_prob = store_prob
        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        self._n_updates = 0
        self._epsilon = 1e-7
        apprx_params_train = deepcopy(approximator_params)
        apprx_params_train['name'] = 'train'
        apprx_params_target = deepcopy(approximator_params)
        apprx_params_target['name'] = 'target'
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

        super(GaussianDQN, self).__init__(policy, mdp_info)

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
                y[:, j] = norm.pdf(x[:, j], loc=mean_list[j], scale=sigma_list[j])
                for k in range(n_actions):
                    if k != j:
                        y[:, j] *= norm.cdf(x[:, j], loc=mean_list[k], scale=sigma_list[k])
                integrals[j] = (upper_limit[j] - lower_limit[j]) / (2 * (n_trapz - 1)) * (
                            y[0, j] + y[-1, j] + 2 * np.sum(y[1:-1, j]))

        # print(np.sum(integrals))
        # assert np.isclose(np.sum(integrals), 1)
        with np.errstate(divide='raise'):
            try:
                return integrals / np.sum(integrals)
            except FloatingPointError:
                print(integrals)
                print(mean_list)
                print(sigma_list)
                input()

    def fit(self, dataset):
        mask = np.ones((len(dataset), 2))
        self._replay_memory.add(dataset,mask)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, mask = \
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next, sigma_next, prob_explore = self._next_q(next_state, absorbing)

            q = reward + self.mdp_info.gamma * q_next
            sigma = self.mdp_info.gamma * sigma_next
            stacked = np.stack([q, sigma])

            self.approximator.fit(state, action, stacked,
                                  prob_exploration=prob_explore,
                                  **self._fit_params)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                `next_state`.

        Returns:
            Maximum action-value for each state in `next_state`.

        """
        q_and_sigma = self.target_approximator.predict(next_state).squeeze()

        q = q_and_sigma[0, :, :]
        sigma = q_and_sigma[1, :, :]
        for i in range(q.shape[0]):
            if absorbing[i]:
                q[i] *= 0
                sigma[i] *= self._epsilon
        max_q = np.zeros((q.shape[0]))
        max_sigma = np.zeros((q.shape[0]))
        probs = []
        '''prob_explore = np.zeros(q.shape[0])
        for i in range(q.shape[0]):  # for each batch
            means = q[i, :]
            sigmas = sigma[i, :]
            prob = GaussianDQN._compute_prob_max(means, sigmas)
            probs.append(prob)
            prob_explore[i] = 1. - np.max(prob)'''
        if self.update_type == 'mean':
            best_actions = np.argmax(q, axis=1)
            for i in range(q.shape[0]):
                max_q[i] = q[i, best_actions[i]]
                max_sigma[i] = sigma[i, best_actions[i]]
        elif self.update_type == 'weighted':
            for i in range(q.shape[0]):  # for each batch
                means = q[i, :]
                sigmas = sigma[i, :]
                prob = probs[i]
                max_q[i] = np.sum(means * prob)
                max_sigma[i] = np.sum(sigmas * prob)
        elif self.update_type == 'optimistic':
            raise ValueError("Optimistic update not implemented")
        else:
            raise ValueError("Update type not implemented")

        return max_q, max_sigma, -1 #np.mean(prob_explore)

    def draw_action(self, state):
        action = super(GaussianDQN, self).draw_action(np.array(state))

        return action

    def episode_start(self):
        return


class GaussianDoubleDQN(GaussianDQN):
    """
    Double DQN algorithm.
    "Deep Reinforcement Learning with Double Q-Learning".
    Hasselt H. V. et al.. 2016.

    """

    def _next_q(self, next_state, absorbing):
        q = np.array(self.approximator.predict(next_state))[0]
        tq = np.array(self.target_approximator.predict(next_state))[0]
        for i in range(q.shape[1]):
            if absorbing[i]:
                tq[:, i, :] *= 1. - absorbing[i]

        max_a = np.argmax(q, axis=2)

        double_q = np.zeros(q.shape[:2])
        for i in range(double_q.shape[0]):
            for j in range(double_q.shape[1]):
                double_q[i, j] = tq[i, j, max_a[i, j]]

        return double_q.T

