import numpy as np
import time


def eval_atari(env, pi, n_timesteps, verbose=True):

    rewards = []
    s = env.reset()
    t = 0
    rew = 0
    start = time.time()

    for i in range(n_timesteps):
        a = pi(s)
        ns, r, done, inf = env.step(a)
        s = ns
        rew += r
        t += 1
        if done:

            if verbose:
                print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
            rewards.append(rew)
            s = env.reset()
            t = 0
            rew = 0
            start = time.time()

    if not done:
        if verbose:
            print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
        rewards.append(rew)

    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))

    env.reset()

    return avg, std
