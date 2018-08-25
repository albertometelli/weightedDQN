import argparse
import os
import sys

from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf

from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import LinearDecayParameter, Parameter

sys.path.append('..')
sys.path.append('../..')
from particle_dqn import ParticleDQN
from policy import BootPolicy, WeightedPolicy
from net import SimpleNet


"""
This script can be used to run Atari experiments with DQN.

"""

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset, gamma):
    J = np.mean(compute_J(dataset, gamma))
    print('J: %f' % J)

    return J


def experiment(policy, name, folder_name):
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()

    arg_mdp = parser.add_argument_group('Environment')
    arg_mdp.add_argument("--horizon", type=int)
    arg_mdp.add_argument("--gamma", type=float)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=100,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=5000,
                         help='Max size of the replay memory.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--n-features", type=int, default=80)
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate", type=float, default=.0001,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered')
    arg_net.add_argument("--epsilon", type=float, default=.01,
                         help='Epsilon term used in rmspropcentered')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--weighted-update", action='store_true')
    arg_alg.add_argument("--n-approximators", type=int, default=10,
                         help="Number of approximators used in the ensemble for"
                              "Averaged DQN.")
    arg_alg.add_argument("--batch-size", type=int, default=100,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=100,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=1000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--train-frequency", type=int, default=1,
                         help='Number of learning steps before each fit of the'
                              'neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=1,
                         help='Number of steps until the exploration rate stops'
                              'decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=0.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=0.,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=1000,
                         help='Number of steps for each evaluation.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--load-path', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')

    args = parser.parse_args()

    scores = list()

    # Evaluation of the model provided by the user.
    if args.load_path:
        # MDP
        if name != 'Taxi':
            mdp = Gym(name, args.horizon, args.gamma)
            n_states = None
            gamma_eval = 1.
        else:
            mdp = generate_taxi('../../grid.txt')
            n_states = mdp.info.observation_space.size[0]
            gamma_eval = mdp.info.gamma

        # Policy
        epsilon_test = Parameter(value=args.test_exploration_rate)
        pi = BootPolicy(args.n_approximators, epsilon=epsilon_test)

        # Approximator
        input_shape = mdp.info.observation_space.shape + (1,)
        input_preprocessor = list()
        approximator_params = dict(
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_states=n_states,
            n_actions=mdp.info.action_space.n,
            n_features=args.n_features,
            n_approximators=args.n_approximators,
            input_preprocessor=input_preprocessor,
            name='test',
            load_path=args.load_path,
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'decay': args.decay,
                       'epsilon': args.epsilon}
        )

        approximator = SimpleNet

        # Agent
        algorithm_params = dict(
            batch_size=0,
            initial_replay_size=0,
            max_replay_size=0,
            clip_reward=False,
            n_approximators=args.n_approximators,
            train_frequency=1,
            target_update_frequency=1,
            weighted_update=args.weighted_update
        )
        agent = ParticleDQN(approximator, pi, mdp.info,
                          approximator_params=approximator_params,
                          **algorithm_params)

        # Algorithm
        core_test = Core(agent, mdp)

        # Evaluate model
        pi.set_eval(True)
        dataset = core_test.evaluate(n_steps=args.test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        get_stats(dataset, gamma_eval)
    else:
        # DQN learning run

        # Settings
        if args.debug:
            initial_replay_size = 50
            max_replay_size = 500
            train_frequency = 5
            target_update_frequency = 10
            test_samples = 20
            evaluation_frequency = 50
            max_steps = 1000
        else:
            initial_replay_size = args.initial_replay_size
            max_replay_size = args.max_replay_size
            train_frequency = args.train_frequency
            target_update_frequency = args.target_update_frequency
            test_samples = args.test_samples
            evaluation_frequency = args.evaluation_frequency
            max_steps = args.max_steps

        # MDP
        if name != 'Taxi':
            mdp = Gym(name, args.horizon, args.gamma)
            n_states = None
            gamma_eval = 1.
        else:
            mdp = generate_taxi('../../grid.txt')
            n_states = mdp.info.observation_space.size[0]
            gamma_eval = mdp.info.gamma

        # Policy
        epsilon = LinearDecayParameter(value=args.initial_exploration_rate,
                                       min_value=args.final_exploration_rate,
                                       n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1.)

        if policy == 'boot':
            pi = BootPolicy(args.n_approximators, epsilon=epsilon_random)
        elif policy == 'weighted':
            pi = WeightedPolicy(args.n_approximators, epsilon=epsilon_random)
        else:
            raise ValueError

        # Approximator
        input_shape = mdp.info.observation_space.shape + (1,)
        input_preprocessor = list()
        approximator_params = dict(
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_states=n_states,
            n_actions=mdp.info.action_space.n,
            n_features=args.n_features,
            n_approximators=args.n_approximators,
            input_preprocessor=input_preprocessor,
            folder_name=folder_name,
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'decay': args.decay,
                       'epsilon': args.epsilon}
        )

        approximator = SimpleNet

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            clip_reward=False,
            n_approximators=args.n_approximators,
            train_frequency=train_frequency,
            target_update_frequency=target_update_frequency,
            weighted_update=args.weighted_update
        )

        agent = ParticleDQN(approximator, pi, mdp.info,
                          approximator_params=approximator_params,
                          **algorithm_params)

        # Algorithm
        core = Core(agent, mdp)
        core_test = Core(agent, mdp)

        # RUN

        # Fill replay memory with random dataset
        print_epoch(0)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.approximator.model.save()

        # Evaluate initial policy
        pi.set_eval(True)
        pi.set_epsilon(epsilon_test)
        dataset = core_test.evaluate(n_steps=test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        scores.append(get_stats(dataset, gamma_eval))

        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch)
            print('- Learning:')
            pi.set_eval(False)
            pi.set_epsilon(epsilon)
            # learning step
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency,
                       quiet=args.quiet)

            if args.save:
                agent.approximator.model.save()

            print('- Evaluation:')
            # evaluation step
            core_test.reset()
            pi.set_eval(True)
            pi.set_epsilon(epsilon_test)
            dataset = core_test.evaluate(n_steps=test_samples,
                                         render=args.render,
                                         quiet=args.quiet)
            scores.append(get_stats(dataset, gamma_eval))

    return scores


if __name__ == '__main__':
    policy = ['boot', 'weighted']
    name = 'Acrobot-v1'

    n_experiments = 10

    for p in policy:
        folder_name = './logs/' + p + '/' + name
        out = Parallel(n_jobs=-1)(
            delayed(experiment)(p, name,
                                folder_name) for _ in range(n_experiments))
        tf.reset_default_graph()

        np.save(folder_name + '/scores.npy', out)
