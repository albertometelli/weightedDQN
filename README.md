# Wasserstein Q-learning

This repository contains the implementation of the WQL algorithm both in tabular domains and Atari games, together with implementation of algorithms we used to confront our results. It uses the [Mushroom](https://github.com/AIRLab-POLIMI/mushroom) RL library and OpenAI Gym.

## Available algorithms
- [P-WQL](q_learning/particle_q_learning.py)
- [G-WQL](q_learning/wq_learning.py)
- [Delayed QL](q_learning/delayed_q_learning.py)
- [Bootstrapped QL](q_learning/boot_q_learning.py)
- [Rmax](r_max/r_max.py)
- [Particle-DQN](dqn/particle_dqn.py)
- [Gaussian-DQN](dqn/gaussian_dqn.py)
- [DQN](dqn/dqn.py)
- [Bootstrapped DQN](dqn/bootstrapped_dqn.py)

## Reproducibility 
To reproduce our results run the following scripts:
- `q_learning/run.py` : Run experiments in tabular RL. You can specify the environments, algorithms, policies, update methods and any other hyperparameter of each algorithm used. By default will run all algorithms in all the environments and log the results in the 'tabular_data' directory.
- `dqn/atari/run.py.py` : Run experiments in atari games. You can specify the environments, algorithms, policies, update methods and any other hyperparameter of each algorithm used. By default will run particle DQN in Breakout using posterior sampling policy with MO update and log the results in the 'logs' directory.
