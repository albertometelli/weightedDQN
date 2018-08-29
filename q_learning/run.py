import sys
import os
import argparse
import numpy as np
import warnings
import time
import random
from joblib import Parallel, delayed
from distutils.util import strtobool

from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.environments.gym_env import Gym
from mushroom.utils.callbacks import CollectDataset, CollectQ
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter
from mushroom.policy.td_policy import EpsGreedy, Boltzmann
from mushroom.algorithms.value.td import QLearning, DoubleQLearning
from mushroom.utils.table import Table

from boot_q_learning import BootstrappedQLearning,  BootstrappedDoubleQLearning
from particle_q_learning import ParticleQLearning, ParticleDoubleQLearning
sys.path.append('..')
from policy import BootPolicy, WeightedPolicy, VPIPolicy
from envs.knight_quest import KnightQuest

from envs.chain import generate_chain
from envs.loop import generate_loop
from envs.river_swim import generate_river
from envs.six_arms import generate_arms

policy_dict = {'eps-greedy': EpsGreedy,
               'boltzmann': Boltzmann,
               'weighted': WeightedPolicy,
               'boot': BootPolicy,
               'vpi': VPIPolicy}


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def compute_scores(dataset, gamma):
    scores = list()
    disc_scores = list()
    lens = list()

    score = 0.
    disc_score = 0.
    episode_steps = 0
    n_episodes = 0
    for i in range(len(dataset)):
        score += dataset[i][2]
        disc_score += dataset[i][2] * gamma ** episode_steps
        episode_steps += 1
        if dataset[i][-1]:
            scores.append(score)
            disc_scores.append(disc_score)
            lens.append(episode_steps)
            score = 0.
            disc_score = 0.
            episode_steps = 0
            n_episodes += 1

    if len(scores) > 0:
        return len(dataset), np.min(scores), np.max(scores), np.mean(scores), np.std(scores),  \
               np.min(disc_scores), np.max(disc_scores), np.mean(disc_scores), \
               np.std(disc_scores), np.mean(lens), n_episodes
    else:
        return len(dataset), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


def experiment(algorithm, name, update_mode, update_type, policy, n_approximators, q_max, q_min, lr_exp, double, seed):
    set_global_seeds(seed)
    print('Using seed %s' % seed)

    # MDP
    if name == 'Taxi':
        mdp = generate_taxi('../grid.txt', horizon=5000)
        max_steps = 500000
        evaluation_frequency = 5000
        test_samples = 5000
    elif name == 'Chain':
        mdp = generate_chain(horizon=100)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'Loop':
        mdp = generate_loop(horizon=100)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'RiverSwim':
        mdp = generate_river(horizon=100)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'SixArms':
        mdp = generate_arms(horizon=100)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'KnightQuest':
        mdp = Gym('KnightQuest-v0', gamma=0.99, horizon=10000)
        max_steps = 500000
        evaluation_frequency = 5000
        test_samples = 5000
    else:
        raise NotImplementedError

    epsilon_test = Parameter(0)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=lr_exp,
                                              size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)

    if algorithm == 'ql':
        if policy not in ['boltzmann', 'eps-greedy']:
            warnings.warn('QL available with only boltzmann and eps-greedy policies!')
            policy = 'eps-greedy'

        if policy == 'eps-greedy':
            epsilon_train = ExponentialDecayParameter(value=1., decay_exp=.5,
                                                      size=mdp.info.observation_space.size)
            pi = policy_dict[policy](epsilon=epsilon_train)
        else:
            beta_train = ExponentialDecayParameter(value=1.5 * q_max, decay_exp=.5,
                                                      size=mdp.info.observation_space.size)
            pi = policy_dict[policy](beta=beta_train)
        if double:
            agent = DoubleQLearning(pi, mdp.info, **algorithm_params)
        else:
            agent = QLearning(pi, mdp.info, **algorithm_params)
    elif algorithm == 'boot-ql':
        if policy not in ['boot', 'weighted']:
            warnings.warn('Bootstrapped QL available with only boot and weighted policies!')
            policy = 'boot'
        pi = policy_dict[policy](n_approximators=n_approximators)
        algorithm_params = dict(n_approximators=n_approximators,
                                mu=(q_max + q_min) / 2,
                                sigma=q_max - q_min,
                                **algorithm_params)
        if double:
            agent = BootstrappedDoubleQLearning(pi, mdp.info, **algorithm_params)
        else:
            agent = BootstrappedQLearning(pi, mdp.info, **algorithm_params)
        epsilon_train = Parameter(0)
    elif algorithm == 'particle-ql':
        if policy not in ['weighted', 'vpi']:
            warnings.warn('Particle QL available with only vpi and weighted policies!')
            policy = 'weighted'
        pi = policy_dict[policy](n_approximators=n_approximators)
        algorithm_params = dict(n_approximators=n_approximators,
                                update_mode=update_mode,
                                update_type=update_type,
                                q_max=q_max,
                                q_min=q_min,
                                **algorithm_params)
        if double:
            agent = ParticleDoubleQLearning(pi, mdp.info, **algorithm_params)
        else:
            agent = ParticleQLearning(pi, mdp.info, **algorithm_params)
        epsilon_train = Parameter(0)
    else:
        raise ValueError()

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    train_scores = []
    test_scores = []

    for n_epoch in range(1, max_steps // evaluation_frequency + 1):

        # Train
        if hasattr(pi, 'set_epsilon'):
            pi.set_epsilon(epsilon_train)
        if hasattr(pi, 'set_eval'):
            pi.set_eval(False)
        core.learn(n_steps=evaluation_frequency, n_steps_per_fit=1, quiet=True)
        dataset = collect_dataset.get()
        scores = compute_scores(dataset, mdp.info.gamma)

        #print('Train: ', scores)
        train_scores.append(scores)

        collect_dataset.clean()
        mdp.reset()

        if hasattr(pi, 'set_epsilon'):
            pi.set_epsilon(epsilon_test)
        if hasattr(pi, 'set_eval'):
            pi.set_eval(True)
        dataset = core.evaluate(n_steps=test_samples, quiet=True)
        mdp.reset()
        scores = compute_scores(dataset, mdp.info.gamma)
        #print('Evaluation: ', scores)
        test_scores.append(scores)

    return train_scores, test_scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    algorithms = ['ql', 'boot-ql', 'particle-ql']
    update_types = ['mean', 'weighted']
    envs = [ "Chain","Taxi", "KnightQuest", "Loop", "RiverSwim", "SixArms"]
    alg_to_policies = {
        "particle-ql": ["weighted", "vpi"],
        "boot-ql": ["boot", "weighted"],
        "ql": ["boltzmann", "eps-greedy"]
    }
    alg_to_update_types={
        "particle-ql": [ "weighted", "mean"],
        "boot-ql": ["weighted"],
        "ql": ["weighted"]
    }
    env_to_qs={
        "KnightQuest": (-20, 20),
        "Taxi": (0, 15),
        "Loop": (0, 40),
        "Chain": (0, 400),
        "RiverSwim": (0, 70000),
        "SixArms": (0, 200000)
    }
    double_vec = [False, True]
    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          choices=[ 
                          "Chain",
                          "Taxi",
                          "KnightQuest",
                          "Loop", 
                          "RiverSwim",
                          "SixArms", 
                          ""],
                          default='',
                          help='Name of the environment to test.')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm",
                         choices=[
                         'ql',
                         'boot-ql',
                         'particle-ql',
                         ''],
                         default='',
                         help='The algorithm.')
    arg_alg.add_argument("--update-mode",
                          choices=['deterministic', 'randomized'],
                          default='deterministic',
                          help='Whether to perform randomized or deterministic target update (only ParticleQLearning).')
    arg_alg.add_argument("--update-type",
                          choices=['mean', 'distributional', 'weighted'],
                          default='weighted',
                          help='Kind of update to perform (only ParticleQLearning).')
    arg_alg.add_argument("--policy",
                          choices=['weighted', 'vpi', 'boot', 'boltzmann', 'eps-greedy'],
                          default='weighted',
                          help='Kind of policy to use (not all available for all).')
    arg_alg.add_argument("--n-approximators", type=int, default=20,
                         help="Number of approximators used in the ensemble.")
    arg_alg.add_argument("--q-max", type=float, default=40,
                         help='Upper bound for initializing the heads of the network (only ParticleQLearning).')
    arg_alg.add_argument("--q-min", type=float, default=0,
                         help='Lower bound for initializing the heads of the network (only ParticleQLearning).')
    arg_alg.add_argument("--lr-exp", type=float, default=0.2,
                         help='Exponential decay for lr')
    arg_alg.add_argument("--double", type=str, default='',
                         help='Whether to use double.')
    arg_run = parser.add_argument_group('Run')
    arg_run.add_argument("--n-experiments", type=int, default=1,
                         help='Number of experiments to execute.')
    arg_run.add_argument("--dir", type=str, default='./data',
                         help='Directory where to save data.')
    arg_run.add_argument("--seed", type=int, default=0,
                         help='Seed.')

    args = parser.parse_args()
    n_experiment = args.n_experiments
    
    affinity = len(os.sched_getaffinity(0))
    if args.name != '':
        envs = [args.name]
    if args.algorithm != '':
        algorithms = [args.algorithm]
    if args.double != '':
        double_vec = [bool(strtobool(args.double))]
    for env in envs:
        for alg in algorithms:
            for policy in alg_to_policies[alg]:     
                for update_type in alg_to_update_types[alg]:
                    for double in double_vec:
                        print('Env: %s - Alg: %s - Policy: %s - Update: %s - Double: %s' % (env, alg, policy, update_type, double))
                        qs = env_to_qs[env]
                        fun_args = [alg, env, args.update_mode, update_type, policy, args.n_approximators, qs[1], qs[0], args.lr_exp, double]
                        out = Parallel(n_jobs=affinity)(delayed(experiment)(*(fun_args + [args.seed+i])) for i in range(n_experiment))
                        out_dir = args.dir + '/' + env + '/' + alg
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        file_name = 'results_%s_%s_%s_%s_double=%s_%s' % (policy, '1' if args.algorithm == 'ql' else args.n_approximators,
                                             '' if alg != 'particle-ql' else update_type, args.lr_exp, double, time.time())
                        np.save(out_dir + '/' + file_name, out)
