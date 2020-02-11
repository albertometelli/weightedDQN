import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from value_iteration.value_iteration import ValueIteration
sys.path.append('..')
from envs.wide_narrow import WideNarrow

p = 0.1
n = 1
w = 6
horizon = 500
mc_sims = 200
gamma = 0.99
n_models = 8
n_states = n *2 +1
n_actions = w
data = np.zeros(shape=(n_models, n_states, n_actions, mc_sims))


def opt_pol(s):
    return 0


for i in range(n_models):
    env = WideNarrow(n=n, w=w, horizon=horizon, gamma=gamma)
    vi = ValueIteration(env, discount_factor=gamma, horizon=horizon)
    vi.fit()
    q_func = vi.get_q_function()
    v_func = vi.get_v_function()
    for state in range(env._mdp_info.observation_space.size[0]):
        print("Doing state %d n model %d" %(state, i))
        for a in range(env._mdp_info.action_space.size[0]):
            for j in range(mc_sims):
                s = env.reset([state])
                s, r, done, inf = env.step([a])
                t = 1
                ret = r

                data[i, state, a, j] = r + gamma * v_func[s]
    p += 0.05

np.save("chain_data_wn_td.npy", data)
data = np.load('chain_data_wn_td.npy')


fig, ax = plt.subplots(n_models, n_states, figsize=(20, 20))
p = 0.1
for i in range(n_models):
    env = WideNarrow(n=n, w=w, horizon=horizon, gamma=gamma)
    vi = ValueIteration(env, discount_factor=gamma, horizon=horizon)
    vi.fit()
    q_func = vi.get_q_function()

    for j in range(n_states):
        xs = data[i, j, 0, :]
        ys = data[i, j, 1, :]
        ax[i, j].scatter(xs, ys, s=5)
        ax[i, j].scatter(xs.mean(), ys.mean(), marker='x', s=40, color='red')
        ax[i, j].scatter([q_func[j, 0]], [q_func[j, 1]], marker='x', s=40, color='black')
        ax[i, j].plot([50,800], [50, 800], ls="--", c=".3")
        x0, x1 = ax[i, j].get_xlim()
        y0, y1 = ax[i, j].get_ylim()
        ax[i, j].set_aspect(abs(x1 - x0) / abs(y1 - y0))
    p += 0.05
plt.show()
