import random
from collections import defaultdict
import numpy as np
# Local classes.
from mushroom.algorithms.agent import Agent

class MBIE_EB(Agent):
    '''
    Implementation for an R-Max Agent [Brafman and Tennenholtz 2003]
    '''

    def __init__(self, mdp_info, rmax=1.0, C=0.03, m=np.inf, tolerance=0.005, value_iterations=10000, epsilon=1):
        #name = name + str(horizon) if name[-2:] == "-h" else name
        super().__init__(self, mdp_info)
        self.rmax = rmax
        self.gamma = mdp_info.gamma
        self.m = m
        self.beta = C * rmax
        self.actions = list(range(mdp_info.size[-1]))
        self.epsilon = epsilon
        self.value_iterations = value_iterations
        self.tolerance = tolerance * rmax
        self.rewards = np.zeros(self.mdp_info.size)
        self.absorbing = defaultdict(lambda: False)  # S --> absorbing
        self.transitions = np.zeros((self.mdp_info.size) + (self.mdp_info.size[0],))  # S --> A --> S' --> counts
        # self.r_s_a_counts = np.zeros(self.mdp_info.size)  # S --> A --> #rs
        self.n_s_a_counts = np.zeros(self.mdp_info.size)  # S --> A --> #ts
        self.n_s_a_s_counts = np.zeros((self.mdp_info.size) + (self.mdp_info.size[0],))  # S --> A --> #ts
        self.q = np.ones(self.mdp_info.size) * (self.rmax / (1. - self.gamma))

        self.prev_state = None
        self.prev_action = None

    def reset(self):
        pass

    def draw_action(self, state):

        # Compute best action.
        state = state[0]

        qs = self.q[state, :]

        action = np.random.choice(np.argwhere(qs == np.max(qs)).ravel())
        return np.array([action])

    def fit(self, dataset):
        assert len(dataset) == 1

        state, action, reward, next_state, absorbing = self._parse(dataset)

        self._update(state, action, reward, next_state, absorbing)

    def _update(self, state, action, reward, next_state, absorbing):

        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)
        Summary:
            Updates T and R.
        '''
        if absorbing:
            self.absorbing[next_state] = True
        nS, nA = self.mdp_info.size
        V = np.zeros(nS)
        policy = {s: None for s in range(nS)}
        if self.n_s_a_counts[state, action] < self.m:
            self.n_s_a_counts[state, action] = self.n_s_a_counts[state, action] + 1
            self.rewards[state, action] = self.rewards[state, action] + reward
            self.transitions[state, action, next_state] = self.transitions[state, action, next_state] + 1

            #run value_iteration
            ite = 0
            delta = np.inf
            while ite < self.value_iterations and delta > self.tolerance:
                delta = 0
                for s in range(self.mdp_info.size[0]):
                    v_val = -np.inf
                    best_action = None
                    for a in range(self.mdp_info.size[1]):
                        if self.n_s_a_counts[s, a] == 0:
                            if v_val < self.q[s, a]:
                                v_val = max(v_val, self.q[s, a])
                                best_action = a
                            continue
                        R = self.rewards[s, a] / self.n_s_a_counts[s, a]
                        v = 0
                        for s1 in range(self.mdp_info.size[0]):
                            if not self.absorbing[s1]:
                                p = self.transitions[s, a, s1] / (self.n_s_a_counts[s, a])
                                v += p * np.max(self.q[s1, :])
                        self.q[s, a] = R + self.mdp_info.gamma * v + (self.beta / np.sqrt(self.n_s_a_counts[s, a]))
                        if v_val < self.q[s, a]:
                            v_val = max(v_val, self.q[s, a])
                            best_action = a

                    if v_val != -np.inf:
                        delta = max(delta, abs(V[s] - v_val))
                        V[s] = v_val
                        policy[s] = best_action
                ite += 1
            self.policy = policy
            
    @staticmethod
    def _parse(dataset):
        """
        Utility to parse the dataset that is supposed to contain only a sample.
        Args:
            dataset (list): the current episode step.
        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag.
        """
        sample = dataset[0]
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]

        return state[0], action[0], reward, next_state[0], absorbing

    def episode_start(self):

        pass

