import sys
import os
import argparse
import numpy as np
import warnings
import time
import random
from joblib import Parallel, delayed
from distutils.util import strtobool
from scipy.stats import norm
from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.environments.gym_env import Gym
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter
from mushroom.policy.td_policy import EpsGreedy, Boltzmann
from mushroom.algorithms.value.td import QLearning, DoubleQLearning
from mushroom.utils.table import Table
from boot_q_learning import BootstrappedQLearning,  BootstrappedDoubleQLearning
from particle_q_learning import ParticleQLearning, ParticleDoubleQLearning
from wq_learning import GaussianQLearning, GaussianDoubleQLearning
from delayed_q_learning import DelayedQLearning
sys.path.append('..')
from policy import BootPolicy, WeightedPolicy, VPIPolicy, WeightedGaussianPolicy, UCBPolicy
from parameter import LogarithmicDecayParameter
from r_max.r_max import RMaxAgent
from mbie.mbie import MBIE_EB
from envs.knight_quest import KnightQuest
from envs.gridworld import generate_gridworld
from envs.chain import generate_chain
from envs.loop import generate_loop
from envs.river_swim import generate_river
from envs.six_arms import generate_arms
from envs.three_arms import generate_arms as generate_three_arms
import envs.knight_quest
from gym.envs.registration import register
from utils.callbacks import CollectQs, CollectVs

policy_dict = {'eps-greedy': EpsGreedy,
               'boltzmann': Boltzmann,
               'weighted': WeightedPolicy,
               'boot': BootPolicy,
               'vpi': VPIPolicy,
               'weighted-gaussian': WeightedGaussianPolicy}


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

class TheoreticalParameter(Parameter):

    def __init__(self, value=1.1, b=2, decay_exp=1., min_value=None, size=(1,)):
        self._decay_exp = decay_exp
        self.a = value
        self.b = b
        super(TheoreticalParameter, self).__init__(value/b, min_value, size)

    def _compute(self, *idx, **kwargs):
        n = np.maximum(self._n_updates[idx], 1)
        return self.a / (self.b + n ** self._decay_exp)

class BetaParameter(Parameter):

    def __init__(self, b=2, min_value=None, size=(1,)):
        self.b = b
        value = 1 - np.sqrt(1 - 1 / self.b)
        super(BetaParameter, self).__init__(value, min_value, size)

    def _compute(self, *idx, **kwargs):
        n = np.maximum(self._n_updates[idx], 1)
        return 1 - np.sqrt(1 - 1 / (1 + n))

def experiment(algorithm, name, update_mode, update_type, policy, n_approximators, q_max, q_min,
               lr_exp, R, log_lr, r_max_m, delayed_m, delayed_epsilon, delta, debug, double,
               regret_test, a, b, mbie_C, value_iterations, tolerance, file_name, out_dir,
               collect_qs,  seed):
    set_global_seeds(seed)
    print('Using seed %s' % seed)
    # MDP
    if name == 'Taxi':
        mdp = generate_taxi('../grid.txt', horizon=5000, gamma=0.99)
        max_steps = 500000
        evaluation_frequency = 5000
        test_samples = 5000
    elif name == 'Chain':
        mdp = generate_chain(horizon=100, gamma=0.99)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'Gridworld':
        mdp = generate_gridworld(horizon=100, gamma=0.99)
        max_steps = 500000
        evaluation_frequency = 5000
        test_samples = 1000
    elif name == 'Loop':
        mdp = generate_loop(horizon=100, gamma=0.99)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'RiverSwim':
        mdp = generate_river(horizon=100, gamma=0.99)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
        mbie_C = 0.4
    elif name == 'SixArms':
        mdp = generate_arms(horizon=100, gamma=0.99)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
        mbie_C = 0.8
    elif name == 'ThreeArms':
        horizon = 100
        mdp = generate_three_arms(horizon=horizon, gamma=0.99)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'KnightQuest':
        mdp = None
        try:
            mdp = Gym('KnightQuest-v0', gamma=0.99, horizon=10000)
        except:
            register(
                id='KnightQuest-v0',
                entry_point='envs.knight_quest:KnightQuest',
            )
            mdp = Gym('KnightQuest-v0', gamma=0.99, horizon=10000)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    else:
        raise NotImplementedError

    epsilon_test = Parameter(0)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=lr_exp,
                                              size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    if regret_test:

        max_steps = 100000000
        evaluation_frequency = 1000000
        test_samples = 1000
        if name == 'ThreeArms':
            max_steps = 200000000
            evaluation_frequency = 2000000
            test_samples = 1000
        if debug:
            max_steps = 100000
            evaluation_frequency = 1000
            test_samples = 1000
        
    if algorithm == 'ql':
        if policy not in ['boltzmann', 'eps-greedy']:
            warnings.warn('QL available with only boltzmann and eps-greedy policies!')
            policy = 'eps-greedy'

        if policy == 'eps-greedy':
            epsilon_train = ExponentialDecayParameter(value=1., decay_exp=lr_exp,
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
                                sigma=(q_max - q_min)/2,
                                **algorithm_params)
        if double:
            agent = BootstrappedDoubleQLearning(pi, mdp.info, **algorithm_params)
        else:
            agent = BootstrappedQLearning(pi, mdp.info, **algorithm_params)
        epsilon_train = Parameter(0)
    elif algorithm == 'particle-ql':
        if policy not in ['weighted', 'ucb']:
            warnings.warn('Particle QL available with only ucb and weighted policies!')
            policy = 'weighted'
        if policy == 'ucb':
            pi = UCBPolicy(delta=delta, q_max=R/(1-mdp.info.gamma))
        else:
            pi = policy_dict[policy](n_approximators=n_approximators)
        algorithm_params = dict(n_approximators=n_approximators,
                                update_mode=update_mode,
                                update_type=update_type,
                                q_max=q_max,
                                q_min=q_min,
                                delta=delta,
                                **algorithm_params)
        if double:
            agent = ParticleDoubleQLearning(pi, mdp.info, **algorithm_params)
        else:
            agent = ParticleQLearning(pi, mdp.info, **algorithm_params)

        epsilon_train = Parameter(0)
    elif algorithm == 'r-max':
        thr_1 = int(np.ceil((4 * mdp.info.size[0] * 1.0/(1-mdp.info.gamma) * R )**3))

        algorithm_params = dict(
            rmax=R,
            s_a_threshold=r_max_m
        )
        agent = RMaxAgent(mdp.info, **algorithm_params)
        pi = agent
        epsilon_train = Parameter(0)
    elif algorithm == 'mbie':


        algorithm_params = dict(
            rmax=R,
            C=mbie_C,
            value_iterations=value_iterations,
            tolerance=tolerance
        )
        agent = MBIE_EB(mdp.info, **algorithm_params)

        pi = agent
        epsilon_train = Parameter(0)
    elif algorithm == 'delayed-ql':
        theoretic_m = delayed_m
        if regret_test:
            gamma = mdp.info.gamma
            Vmax = R / (1 - gamma)
            epsilon = 0.26 * Vmax
            delayed_epsilon = epsilon*(1-gamma)
            delta = 0.1
            S, A = mdp.info.size

            theoretic_m = (1 + gamma*Vmax)**2 / (2*delayed_epsilon**2) * np.log(3*S*A/delta * (1 + S*A/(delayed_epsilon*(1-gamma))))
            if debug:
                print("Delta:{}".format(delta))
                print("R:{}".format(R))
                print("Vmax:{}".format(Vmax))
                print("Gamma:{}".format(mdp.info.gamma))
                print("Epsilon:{}".format(epsilon))
                #print("k:{}".format(k))
                print("m:{}".format(theoretic_m))
                print("S:{}".format(S))
                print("A:{}".format(A))
                input()
            def evaluate_policy(P, R, policy):

                P_pi = np.zeros((S, S))
                R_pi = np.zeros(S)

                for s in range(S):
                    for s1 in range(S):
                        P_pi[s,s1] = np.sum(policy[s, :] * P[s, :, s1])
                    R_pi[s] = np.sum(policy[s, :] * np.sum(P[s, :, :] * R[s, :, :], axis=-1))
                I = np.diag(np.ones(S))
                V = np.linalg.solve(I - gamma * P_pi, R_pi)

                return V
        algorithm_params = dict(
            R=R,
            m=theoretic_m,
            delta=delta,
            epsilon=delayed_epsilon,
            **algorithm_params)

        agent = DelayedQLearning(mdp.info, **algorithm_params)
        if regret_test:
            collect_vs_callback = CollectVs(mdp, agent, evaluate_policy, 10000)
            if debug:
                print("Q:")
                print(agent.get_approximator()[:, :])
                print("Policy:")
                print(agent.get_policy())
                print("V:{}".format(evaluate_policy(mdp.p,mdp.r,agent.get_policy())))
                input()

        pi = agent
        epsilon_train = Parameter(0)
    elif algorithm == 'gaussian-ql':
        if policy not in ['weighted-gaussian', 'ucb']:
            warnings.warn('Particle QL available with only ucb and weighted policies!')
            policy = 'weighted-gaussian'
        if policy == 'ucb':
            pi = UCBPolicy(delta=delta, q_max=R/(1-mdp.info.gamma))
        else:
            pi = policy_dict[policy]()
        q_0 = (q_max - q_min) / 2
        sigma_0 = (q_max - q_min) / np.sqrt(12)
        C = 2 * R / (np.sqrt(2 * np.pi) * (1 - mdp.info.gamma) * sigma_0)
        sigma_lr = None
        if log_lr:
            sigma_lr = LogarithmicDecayParameter(value=1., C=C,
                                             size=mdp.info.size)
        init_values = (q_0, sigma_0)
        if regret_test:
            sigma_lr = None
            gamma = mdp.info.gamma
            T = max_steps
            S, A = mdp.info.size
            a = 1 / (1 - gamma) + 1
            b = a - 1
            q_max = R / (1 - gamma)
            standard_bound = norm.ppf(1 - delta, loc=0, scale=1)
            first_fac = np.sqrt(b + T)
            second_fac = np.sqrt(a * np.log(S*A*T / delta))
            sigma2_factor = min(np.sqrt(b + T), np.sqrt(a * np.log(S*A*T / delta)))

            q_0 = q_max
            sigma1_0 = 0
            sigma2_0 = (R + gamma * q_max) / (standard_bound * np.sqrt(b-1)) * sigma2_factor
            init_values = (q_0, sigma1_0, sigma2_0)
            learning_rate = TheoreticalParameter(value=a, b=b, decay_exp=1,
                                                 size=mdp.info.size)
            algorithm_params = dict(learning_rate=learning_rate)
            sigma_lr = BetaParameter(b=b, size=mdp.info.size)
            def evaluate_policy(P, R, policy):

                P_pi = np.zeros((S, S))
                R_pi = np.zeros(S)

                for s in range(S):
                    for s1 in range(S):
                        P_pi[s,s1] = np.sum(policy[s, :] * P[s, :, s1])

                    R_pi[s] = np.sum(policy[s, :] * np.sum(P[s, :, :] * R[s, :, :],axis=-1))
                I = np.diag(np.ones(S))

                V = np.linalg.solve(I - gamma * P_pi, R_pi)
                return V
            if debug:
                print("Delta:{}".format(delta))
                print("R:{}".format(R))
                print("Gamma:{}".format(mdp.info.gamma))
                print("mu0:{}".format(q_0))
                print("Sigma1_0:{}".format(sigma1_0))
                print("Sigma2_0:{}".format(sigma2_0))
                print("T:{}".format(T))
                print("S:{}".format(S))
                print("A:{}".format(A))
                input()




        algorithm_params = dict(
            update_mode=update_mode,
            update_type=update_type,
            sigma_learning_rate=sigma_lr,
            init_values=init_values,
            delta=delta,
            q_max=q_max,
            **algorithm_params)
        if double and not regret_test:
            agent = GaussianDoubleQLearning(pi, mdp.info, **algorithm_params)
        else:
            agent = GaussianQLearning(pi, mdp.info, **algorithm_params)
        if regret_test:
            if debug:
                freq = 10
            else:
                freq = 10000
            collect_vs_callback = CollectVs(mdp, agent, evaluate_policy, freq)
        if debug:
            print("Policy:")
            print(agent.get_policy())
            print("Q")
            for state in range(S):
                means = np.array(agent.approximator.predict(np.array([state]), idx=0))
                sigmas1 = np.array(agent.approximator.predict(np.array([state]), idx=1))
                sigmas2 = np.array(agent.approximator.predict(np.array([state]), idx=2))
                print("Means:{}".format(means))
                print("Sigmas1:{}".format(sigmas1))
                print("Sigmas2:{}".format(sigmas2))
            print("V:{}".format(evaluate_policy(mdp.p,mdp.r,agent.get_policy())))
            input()
        if policy == 'ucb':
            q = agent.approximator
            standard_bound = norm.ppf(1 - delta, loc=0, scale=1)
            def quantile_func(state):
                means = np.array(q.predict(state, idx=0))
                if regret_test:
                    sigmas1 = np.array(q.predict(state, idx=1))
                    sigmas2 = np.array(q.predict(state, idx=2))
                    sigmas = sigmas1 + sigmas2
                else:
                    sigmas = np.array(q.predict(state, idx=1))
                out = sigmas * standard_bound + means
                return out

            def mu(state):
                q_list = q.predict(state, idx=0)
                means = np.array(q_list)

                return means
            pi.set_quantile_func(quantile_func)
            pi.set_mu(mu)
        epsilon_train = Parameter(0)
    else:
        raise ValueError()

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    if collect_qs:
        if algorithm not in ['r-max']:
            collect_qs_callback = CollectQs(agent.approximator)
            callbacks += [collect_qs_callback]

    if regret_test:
        callbacks += [collect_vs_callback]
    core = Core(agent, mdp, callbacks)

    train_scores = []
    test_scores = []

    for n_epoch in range(1, max_steps // evaluation_frequency + 1):

        # Train
        if hasattr(pi, 'set_epsilon'):
            pi.set_epsilon(epsilon_train)
        if hasattr(pi, 'set_eval'):
            pi.set_eval(False)
        if regret_test:
            collect_vs_callback.on()
        core.learn(n_steps=evaluation_frequency, n_steps_per_fit=1, quiet=True)
        dataset = collect_dataset.get()
        scores = compute_scores(dataset, mdp.info.gamma)

        #print('Train: ', scores)
        train_scores.append(scores)

        collect_dataset.clean()
        mdp.reset()
        if regret_test:
            vs = collect_vs_callback.get_values()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Finished {} steps.".format(n_epoch * evaluation_frequency))
            np.save(out_dir + "/vs_" + algorithm+"_"+str(seed), vs)
            np.save(out_dir+"/scores_online" + str(seed), train_scores)
            collect_vs_callback.off()
        if hasattr(pi, 'set_epsilon'):
            pi.set_epsilon(epsilon_test)
        if hasattr(pi, 'set_eval'):
            pi.set_eval(True)
        dataset = core.evaluate(n_steps=test_samples, quiet=True)
        mdp.reset()
        scores = compute_scores(dataset, mdp.info.gamma)
        print('Evaluation #%d:%s ' %(n_epoch, scores))

        test_scores.append(scores)
        if regret_test:
            np.save(out_dir + "/scores_offline" + str(seed), test_scores)
    if collect_qs:
        qs= collect_qs_callback.get_values()
        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
        np.save(out_dir + '/' + file_name, qs)

    return train_scores, test_scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    algorithms = ['ql', 'boot-ql', 'particle-ql', 'gaussian-ql','r-max', 'delayed-ql']
    update_types = ['mean', 'weighted']
    envs = ["Gridworld","RiverSwim", "SixArms", "Chain", "Taxi", "KnightQuest", "Loop", "ThreeArms"]
    alg_to_policies = {
        "particle-ql": ["weighted", "ucb"],#, "vpi"
        "boot-ql": ["boot", "weighted"],
        "ql": ["boltzmann", "eps-greedy"],
        "gaussian-ql": ["weighted-gaussian","ucb"],
        "r-max": ["rmax"],
        "mbie": ["mbie"],
        "delayed-ql": ["greedy"]
    }
    alg_to_double_vec = {
        "particle-ql": [False],  # , "vpi"
        "boot-ql": [ True],
        "ql": [ True],
        "gaussian-ql": [False],
        "r-max": [False],
        "mbie": [False],
        "delayed-ql": [False]
    }
    alg_to_update_types = {
        "particle-ql": [ "weighted", "mean", "optimistic"],
        "boot-ql": ["weighted"],
        "ql": ["weighted"],
        "gaussian-ql": ["weighted", "mean", "optimistic"],
        "r-max": ["weighted"],
        "mbie": ["weighted"],
        "delayed-ql": ["weighted"]
    }
    env_to_qs = {
        "Gridworld": (-1000, 1000),
        "KnightQuest": (-20, 0),
        "Taxi": (0, 15),
        "Loop": (0, 40),
        "Chain": (0, 400),
        "RiverSwim": (0, 70000),
        "SixArms": (0, 600000),
        "ThreeArms":(0,30000)
    }
    env_to_R = {
        "Gridworld": -10,
        "KnightQuest": 1.,
        "Taxi": 7.,
        "Loop": 2.,
        "Chain": 10.,
        "RiverSwim": 10000.,
        "SixArms": 6000.,
        "ThreeArms": 300.
    }

    double_vec = [False, True]

    arg_game = parser.add_argument_group('Game')

    arg_game.add_argument("--name",
                          choices=[
                          "Gridworld",
                          "Chain",
                          "Taxi",
                          "KnightQuest",
                          "Loop",
                          "RiverSwim",
                          "SixArms",
                          "ThreeArms",
                          ""],
                          default='',
                          help='Name of the environment to test.')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm",
                         choices=[
                         'ql',
                         'boot-ql',
                         'particle-ql',
                         'gaussian-ql',
                         'r-max',
                         'delayed-ql',
                         'mbie',
                         ''],
                         default='',
                         help='The algorithm.')
    arg_alg.add_argument("--update-mode",
                          choices=['deterministic', 'randomized'],
                          default='deterministic',
                          help='Whether to perform randomized or deterministic target update (only ParticleQLearning).')
    arg_alg.add_argument("--update-type",
                          choices=['mean', 'weighted', 'optimistic'],
                          default='',
                          help='Kind of update to perform (only WQL algorithms).')
    arg_alg.add_argument("--policy",
                          choices=['weighted', 'vpi', 'boot', 'boltzmann', 'eps-greedy', 'ucb', 'weighted-gaussian'],
                          default='',
                          help='Kind of policy to use (not all available for all).')
    arg_alg.add_argument("--n-approximators", type=int, default=20,
                         help="Number of particles used (Particle QL).")
    arg_alg.add_argument("--horizon", type=int, default=1000,
                         help="Horizon of r-max algorithm.")
    arg_alg.add_argument("--m", type=int, default=1000,
                         help="threshold for r-max algorithm.")
    arg_alg.add_argument("--value_iterations", type=int, default=5000,
                         help="max_iterations of VI")
    arg_alg.add_argument("--tolerance", type=float, default=0.01,
                         help="tolerance of value_iteration")
    arg_alg.add_argument("--C", type=float, default=0.3,
                         help="parameter of MBIE")
    arg_alg.add_argument("--delayed-m", type=float, default=1.0,
                         help="m parameter of delayed-ql.")
    arg_alg.add_argument("--epsilon", type=int, default=1.0,
                         help="threshold for delayed-q algorithm.")
    arg_alg.add_argument("--q-max", type=float, default=40,
                         help='Upper bound for initializing the distributions(only WQL and Boot-QL).')
    arg_alg.add_argument("--q-min", type=float, default=0,
                         help='Lower bound for initializing the distributions (only WQL and Boot-QL).')
    arg_alg.add_argument("--lr-exp", type=float, default=0.2,
                         help='Exponential decay for lr')
    arg_alg.add_argument("--a", type=float, default=1.1,
                         help='numerator for learning rate in the regret test')
    arg_alg.add_argument("--b", type=float, default=2.0,
                         help='shift for denominator for learning rate in the regret test')
    arg_alg.add_argument("--delta", type=float, default=0.1,
                         help='confidence bound parameter')
    arg_alg.add_argument("--double", type=str, default='',
                         help='Whether to use double estimators.')
    arg_alg.add_argument("--log-lr",  action='store_true',
                         help='Whether to use log learning rate for gaussian-ql.')
    arg_alg.add_argument("--regret-test", action='store_true',
                         help='Whether to run the regret tests')
    arg_run = parser.add_argument_group('Run')
    arg_run.add_argument("--n-experiments", type=int, default=10,
                         help='Number of experiments to execute.')
    arg_run.add_argument("--dir", type=str, default='./tabular_data',
                         help='Directory where to save data.')
    arg_run.add_argument("--seed", type=int, default=0,
                         help='Seed.')
    arg_run.add_argument("--collect-qs", action='store_true',
                         help="Whether to collect the q_values for each timestep.")
    arg_run.add_argument("--debug", action='store_true',
                         help="Debug flag for the regret test.")

    args = parser.parse_args()
    n_experiment = args.n_experiments

    affinity = min(len(os.sched_getaffinity(0)), args.n_experiments)
    if args.name != '':
        envs = [args.name]
    if args.algorithm != '':
        algorithms = [args.algorithm]
        if args.policy!='' and args.policy in alg_to_policies[args.algorithm]:
            alg_to_policies[args.algorithm]=[args.policy]
        if args.update_type!='' and args.update_type in alg_to_update_types[args.algorithm]:
            alg_to_update_types[args.algorithm]=[args.update_type]

    if args.double != '':
        double_vec = [bool(strtobool(args.double))]
    if args.algorithm in ['r-max','delayed-ql']:
        double_vec = [False]
    if args.policy in ['ucb']:
        algorithms = ['gaussian-ql', 'particle-ql']
        if args.algorithm != '':
            algorithms = [args.algorithm]
        alg_to_policies['gaussian-ql'] = ['ucb']
        alg_to_policies['particle-ql'] = ['ucb']
        alg_to_policies['particle-ql'] = ['ucb']
    if args.regret_test:
        algorithms = ['gaussian-ql','delayed-ql']
        alg_to_policies['delayed-ql'] = ['weighted']
        alg_to_update_types['delayed-ql'] = ['greedy']
        alg_to_policies['gaussian-ql'] = ['ucb']
        alg_to_update_types['gaussian-ql'] = ['optimistic']
        if args.algorithm in algorithms:
            algorithms = [args.algorithm]
        envs = ['ThreeArms']
        if args.name in ["ThreeArms", "SixArms"]:
            envs = [args.name]
        args.lr_exp = 1
    for alg in algorithms:
        for double in alg_to_double_vec[alg]:
            for env in envs:
                for policy in alg_to_policies[alg]:
                    for update_type in alg_to_update_types[alg]:
                        print('Env: %s - Alg: %s - Policy: %s - Update: %s - Double: %s' % (env, alg, policy, update_type, double))
                        qs = env_to_qs[env]
                        R = env_to_R[env]
                        file_name = 'qs_%s_%s_%s_%s_double=%s_%s' % (policy, '1' if args.algorithm == 'ql' else args.n_approximators,
                                             '' if alg != 'particle-ql' else update_type, args.lr_exp, double, time.time())
                        out_dir = args.dir + '/' + env + '/' + alg
                        fun_args = [alg, env, args.update_mode, update_type, policy, args.n_approximators, qs[1],
                                    qs[0], args.lr_exp, R, args.log_lr, args.m, args.delayed_m, args.epsilon,
                                    args.delta, args.debug, double, args.regret_test, args.a, args.b, args.C,
                                    args.value_iterations, args.tolerance, file_name, out_dir]
                        start = time.time()
                        if n_experiment > 1:
                            out = Parallel(n_jobs=affinity)(delayed(experiment)(*(fun_args + [args.collect_qs if i==0 else False, args.seed+i])) for i in range(n_experiment))
                        else:
                            out = [experiment(*(fun_args + [False, 0]))]
                        end = time.time()
                        print("Executed in %f seconds!" %(end - start))
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        file_name = 'results_%s_%s_%s_%s_double=%s_%s' % (policy, '1' if args.algorithm in ['ql', 'weighted-gaussian'] else args.n_approximators,
                                             '' if alg not in ['particle-ql','gaussian-ql'] else update_type, args.lr_exp, double, time.time())
                        if args.algorithm == 'gaussian-ql':
                            file_name += "_log_lr=%s" % (args.log_lr)
                        np.save(out_dir + '/' + file_name, out)
                        
