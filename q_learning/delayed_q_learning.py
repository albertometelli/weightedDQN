import numpy as np
from copy import deepcopy
from mushroom.algorithms.value import TD
from mushroom.utils.table import EnsembleTable, Table

class DelayedQLearning(TD):
    """
    Delayed Q-Learning algorithm.
    """
    def __init__(self, mdp_info, learning_rate, m=None, epsilon=1, R=1., delta=0.1):
        self.Q = Table(mdp_info.size, initial_value=R / (1 - mdp_info.gamma))

        super().__init__(self.Q, self, mdp_info, learning_rate)
        gamma = self.mdp_info.gamma
        delayed_epsilon = epsilon * (1-gamma)
        Vmax = R / (1-self.mdp_info.gamma)
        S, A = self.mdp_info.size
        if m is None or m == 0:
            self.m = (1 + gamma * Vmax) ** 2 / (2 * delayed_epsilon ** 2) * np.log(
                3 * S * A / delta * (1 + S * A / (delayed_epsilon * (1 - gamma))))
        else:
            self.m = int(max(m, 1))
        self.nS = S
        self.nA = A
        self.epsilon = delayed_epsilon
        self.U = Table(mdp_info.size)
        self.l = Table(mdp_info.size)
        #self.t = Table(mdp_info.size)
        self.b = Table(mdp_info.size)
        self.LEARN = Table(mdp_info.size, initial_value=1)
        self.last_t = 0
        self.update_count = 0
        policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            qs = self.Q[s, :]
            actions = np.argwhere(qs == np.max(qs)).ravel()
            n = len(actions)
            for a in actions:
                policy[s, a] = 1. / n
        self.policy_matrix = policy

    def _update(self, state, action, reward, next_state, absorbing):
        self.update_count += 1
        if self.b[state, action] <= self.last_t:
            self.LEARN[state, action] = 1

        if self.LEARN[state, action] == 1:
            if self.l[state, action] == 0:
                self.b[state, action] = self.update_count

            self.U[state, action] += reward + (self.mdp_info.gamma * np.max(self.Q[next_state, :])
                                               if not absorbing else 0)
            self.l[state, action] += 1

            if self.l[state, action] == self.m:
                if self.Q[state, action] - (self.U[state, action] / self.m) >= 2 * self.epsilon:

                    self.Q[state, action] = (self.U[state, action] / self.m) + self.epsilon
                    self.last_t = self.update_count
                    #Update policy matrix NOT PART OF DELAYED!!
                    qs = self.Q[state, :]
                    actions = np.argwhere(qs == np.max(qs)).ravel()
                    n = len(actions)
                    for a in range(self.mdp_info.size[-1]):
                        if a in actions:
                            self.policy_matrix[state, a] = 1. / n
                        else:
                            self.policy_matrix[state, a] = 0

                elif self.b[state, action] > self.last_t:
                    self.LEARN[state, action] = 0

                self.U[state, action] = 0
                self.l[state, action] = 0

    def draw_action(self, state):

        # Compute best action.
        state = state[0]

        qs = self.Q[state, :]
        action = np.random.choice(np.argwhere(qs == np.max(qs)).ravel())
        return np.array([action])

    def episode_start(self):
        pass

    def set_q(self, approximator):
        pass

    def reset(self):
        pass

    def get_policy(self):

        return self.policy_matrix

    def get_approximator(self):
        return self.Q
    def evaluate_policy(self, P, R, policy):


        P_pi = np.zeros((self.nS, self.nS))
        R_pi = np.zeros((self.nS, self.nA))

        for s in range(self.nS):
            for s1 in range(self.nS):
                P_pi = np.sum(policy[s, :] * P[s, :, s1])
            for a in range(self.nA):
                R_pi = policy[s, a] * np.sum(P[s, a, :] * R[s, a, :])
        I = np.diag(np.ones(self.nS))
        V = np.solve(I - self.mdp_info.gamma * P_pi, R_pi)
        return V