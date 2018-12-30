from copy import deepcopy

import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor

from replay_memory import ReplayMemory


class ParticleDQN(Agent):
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 target_update_frequency, initial_replay_size,
                 max_replay_size, fit_params=None, approximator_params=None,
                 n_approximators=1, clip_reward=True,
                 weighted_update=False):
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_approximators = n_approximators
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency
        self.weighted_update = weighted_update

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        self._n_updates = 0

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

        super(ParticleDQN, self).__init__(policy, mdp_info)
    
    @staticmethod
    def _compute_prob_max(q_list):
        q_array = np.array(q_list).T
        score = (q_array[:, :, None, None] >= q_array).astype(int)
        prob = score.sum(axis=3).prod(axis=2).sum(axis=1)
        prob = prob.astype(np.float32)
        return prob / np.sum(prob)

    def fit(self, dataset):
        
        mask = np.ones((len(dataset), self._n_approximators))
        self._replay_memory.add(dataset, mask)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, mask =\
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next, prob_explore = self._next_q(next_state, absorbing)

            q = reward.reshape(self._batch_size,
                               1) + self.mdp_info.gamma * q_next
            self.approximator.fit(state, action, q, mask=mask,
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
        q = np.array(self.target_approximator.predict(next_state))[0]
        for i in range(q.shape[1]):
            if absorbing[i]:
                q[:, i, :] *= 1. - absorbing[i]

        max_q = np.zeros((q.shape[1], q.shape[0]))
        probs = []
        prob_explore = []
        for i in range(q.shape[1]):  # for each batch
            particles = q[:, i, :]
            prob = ParticleDQN._compute_prob_max(particles)
            probs.append(prob)
            prob_explore.append(1 - np.max(prob))
        prob_explore = np.mean(prob_explore)
        if not self.weighted_update:
            best_actions = np.argmax(np.mean(q, axis=0), axis=1)
            for i in range(q.shape[1]):
                max_q[i, :] = q[:, i, best_actions[i]]
        else:
            for i in range(q.shape[1]): #for each batch 
                particles = q[:, i, :]
                prob = probs[i]
                max_q[i, :] = np.dot(particles, prob)
        return max_q, prob_explore


    def draw_action(self, state):
        action = super(ParticleDQN, self).draw_action(np.array(state))

        return action

    def episode_start(self):
        self.policy.set_idx(np.random.randint(self._n_approximators))


class ParticleDoubleDQN(ParticleDQN):
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

