import random
from collections import defaultdict
import numpy as np
# Local classes.
from mushroom.algorithms.agent import Agent

class RMaxAgent(Agent):
    '''
    Implementation for an R-Max Agent [Brafman and Tennenholtz 2003]
    '''

    def __init__(self, mdp_info, rmax=1.0, s_a_threshold = None, epsilon=1):
        #name = name + str(horizon) if name[-2:] == "-h" else name
        super().__init__(self, mdp_info)
        self.rmax = rmax
        self.gamma = mdp_info.gamma
        if s_a_threshold is not None:
            self.s_a_threshold = s_a_threshold
        else:
            self.s_a_threshold = (4 * mdp_info.size[0] * 1/(1-self.gamma) * rmax)
        self.actions = list(range(mdp_info.size[-1]))
        self.epsilon = epsilon
        self.value_iterations = int(np.log(1/(self.epsilon*(1-self.gamma))) / (1-self.gamma))

        self.rewards = np.zeros(self.mdp_info.size)
        self.absorbing = defaultdict(lambda: False)  # S --> absorbing
        self.transitions = np.zeros((self.mdp_info.size) + (self.mdp_info.size[0],))  # S --> A --> S' --> counts
        # self.r_s_a_counts = np.zeros(self.mdp_info.size)  # S --> A --> #rs
        self.n_s_a_counts = np.zeros(self.mdp_info.size)  # S --> A --> #ts
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

        if self.n_s_a_counts[state, action] < self.s_a_threshold:
            self.n_s_a_counts[state, action] = self.n_s_a_counts[state, action] + 1
            self.rewards[state, action] = self.rewards[state, action] + reward
            self.transitions[state, action, next_state] = self.transitions[state, action, next_state] + 1
            if self.n_s_a_counts[state, action] == self.s_a_threshold:
                print("State:{}, Action:{} Known!!".format(state,action))
                for i in range(self.value_iterations):
                    for s in range(self.mdp_info.size[0]):
                        for a in range(self.mdp_info.size[1]):
                            if self.n_s_a_counts[s, a] >= self.s_a_threshold:
                                R = self.rewards[s,a] / self.n_s_a_counts[s, a]
                                T = np.zeros(self.mdp_info.size[0])
                                v = 0
                                for s1 in range(self.mdp_info.size[0]):
                                    p = self.transitions[s, a, s1] / self.n_s_a_counts[s, a]
                                    if not self.absorbing[s1]:
                                        v += p * np.max(self.q[s1:])
                                self.q[s, a] = R + self.mdp_info.gamma * v


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
