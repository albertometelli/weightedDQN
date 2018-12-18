import argparse
import os
import sys
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
sys.path.append('..')
sys.path.append('../..')
try:
    sys.path.remove('/home/alberto/baselines')
except:
    print("")
from baselines import weighted_deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from eval_policy import eval_atari
import gym

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    act = weighted_deepq.particle_learn.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()