import argparse
import os
import sys
from joblib import Parallel, delayed

import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from mushroom.environments import *
from mushroom.environments.generators.taxi import generate_taxi
sys.path.append('..')
sys.path.append('../..')
from particle_dqn import ParticleDQN, ParticleDoubleDQN
from bootstrapped_dqn import BootstrappedDoubleDQN, BootstrappedDQN
from gaussian_dqn import GaussianDQN
from dqn import DoubleDQN, DQN
from mushroom.core.core import Core
from mushroom.utils.dataset import compute_scores
from mushroom.utils.parameters import LinearDecayParameter, Parameter
from policy import BootPolicy, WeightedPolicy, WeightedGaussianPolicy, EpsGreedy, UCBPolicy
from envs.gridworld import GridWorld
import time
import tensorflow as tf
"""
This script can be used to run Atari experiments with DQN.

"""

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')

def get_stats(dataset):
    score = compute_scores(dataset)
    print('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score)

    return score

def experiment(args, agent_algorithm):
    np.random.seed()



    scores = list()
    #add timestamp to results
    ts=str(time.time())
    # Evaluation of the model provided by the user.
    if args.load_path and args.evaluation:
        # MDP
        if args.name not in ['Taxi','Gridworld']:
            mdp = Gym(args.name, args.horizon, args.gamma)
            n_states = None
            gamma_eval = 1.
        elif args.name == 'Taxi':
            mdp = generate_taxi('../../grid.txt')
            n_states = mdp.info.observation_space.size[0]
            gamma_eval = mdp.info.gamma
        else:
            rew_weights = [args.fast_zone, args.slow_zone, args.goal]
            grid_size = args.grid_size
            env = GridWorld(gamma=args.gamma, rew_weights=rew_weights,
                         shape=(grid_size, grid_size), randomized_initial=args.rand_initial,
                            horizon=args.horizon)
            gamma_eval = args.gamma
            mdp = env.generate_mdp()
            n_states = mdp.info.observation_space.size[0]
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
            net_type=args.net_type,
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'lr_sigma': args.learning_rate,
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
            target_update_frequency=1
        )
        if args.alg == 'boot':
            algorithm_params['p_mask'] = args.p_mask
            pi = BootPolicy(args.n_approximators, epsilon=epsilon_test)
        elif args.alg == 'gaussian':
            if args.ucb:
                pi = UCBPolicy(delta=args.delta, q_max=1. / (1.-args.gamma))
            else:
                pi = WeightedGaussianPolicy(epsilon=epsilon_test)
        elif args.alg == 'dqn':
            pi = EpsGreedy(epsilon=epsilon_test)
        elif args.alg =='particle':
            if args.ucb:
                pi = UCBPolicy(delta=args.delta, q_max=1. / (1.-args.gamma))
            else:
                pi = WeightedPolicy(args.n_approximators, epsilon=epsilon_test)

        else:
            raise ValueError("Algorithm uknown")

        if args.alg in ['gaussian', 'particle']:
            algorithm_params['update_type']=args.update_type
            algorithm_params['delta'] = args.delta
            algorithm_params['store_prob'] = args.store_prob
            if args.clip_target:
                algorithm_params['max_spread'] = args.q_max - args.q_min
            approximator_params['q_min']= args.q_min
            approximator_params['q_max']= args.q_max
            approximator_params['loss']= args.loss
            approximator_params['init_type'] = args.init_type
            approximator_params['sigma_weight'] = args.sigma_weight
        if  args.alg in ['particle', 'boot']:
            approximator_params['n_approximators'] = args.n_approximators
            algorithm_params['n_approximators'] = args.n_approximators
        agent = agent_algorithm(approximator, pi, mdp.info,
                          approximator_params=approximator_params,
                          **algorithm_params)

        # Algorithm
        core_test = Core(agent, mdp)

        # Evaluate model
        pi.set_eval(True)
        dataset = core_test.evaluate(n_steps=args.test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        get_stats(dataset)
    else:
        # DQN learning run
        print("Learning Run")

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
        if args.name not in ['Taxi', 'Gridworld']:
            mdp = Gym(args.name, args.horizon, args.gamma)
            n_states = None
            gamma_eval = 1.
        elif args.name =='Taxi':
            mdp = generate_taxi('../../grid.txt')
            n_states = mdp.info.observation_space.size[0]
            gamma_eval = mdp.info.gamma
        else:
            rew_weights = [args.fast_zone, args.slow_zone, args.goal]
            grid_size = args.grid_size
            env = GridWorld(gamma=args.gamma, rew_weights=rew_weights,
                            shape=(grid_size, grid_size), randomized_initial=args.rand_initial,
                            horizon=args.horizon)
            mdp = env.generate_mdp()
            n_states = mdp.info.observation_space.size[0]
            print(mdp.info.gamma)
            gamma_eval = args.gamma
        # Policy
        epsilon = LinearDecayParameter(value=args.initial_exploration_rate,
                                       min_value=args.final_exploration_rate,
                                       n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1.)

        policy_name = 'weighted'
        update_rule = args.update_type + "_update"
        if args.alg == 'boot':
            pi = BootPolicy(args.n_approximators, epsilon = epsilon)
            policy_name = 'boot'
            update_rule = 'boot'
        elif args.alg == 'dqn':
            pi = EpsGreedy(epsilon=epsilon)
            policy_name = 'eps_greedy'
            update_rule = 'td'
        elif args.alg == 'particle':
            if args.ucb:
                policy_name = 'ucb'
                pi = UCBPolicy(delta=args.delta, q_max=1. / (1.-args.gamma))
            else:
                pi = WeightedPolicy(args.n_approximators)
        elif args.alg == 'gaussian':
            if args.ucb:
                policy_name = 'ucb'
                pi = UCBPolicy(delta=args.delta, q_max=1. / (1.-args.gamma))
            else:
                pi = WeightedGaussianPolicy()
        else:
            raise ValueError("Algorithm unknown")
        # Summary folder
        folder_name = './logs/' + args.alg + "/" + policy_name + '/' + update_rule + '/' + args.name + "/" + args.loss + "/" + str(
            args.n_approximators) + "_particles" + "/" + args.init_type + "_init" + "/" + str(
            args.learning_rate) + "/" + ts

        # Approximator
        input_shape = mdp.info.observation_space.shape
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
            net_type=args.net_type,
            sigma_weight=args.sigma_weight,
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'lr_sigma': args.learning_rate,
                       'decay': args.decay,
                       'epsilon': args.epsilon}
        )
        if args.load_path:
            ts=os.path.basename(os.path.normpath(args.load_path))
            approximator_params['load_path']=args.load_path
            approximator_params['folder_name']=args.load_path
            folder_name=args.load_path
            p = "scores_"+str(ts)+".npy"
            scores=np.load(p).tolist()
            max_steps=max_steps-evaluation_frequency*len(scores)
        approximator = SimpleNet

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            clip_reward=False,
            target_update_frequency=target_update_frequency // train_frequency,
        )
        if args.alg == 'boot':
            algorithm_params['p_mask']=args.p_mask
        elif args.alg in ['particle', 'gaussian']:
            algorithm_params['update_type'] = args.update_type
            algorithm_params['delta'] = args.delta
            algorithm_params['store_prob'] = args.store_prob
            if args.clip_target:
                algorithm_params['max_spread'] = args.q_max - args.q_min
            approximator_params['q_min'] = args.q_min
            approximator_params['q_max'] = args.q_max
            approximator_params['loss'] = args.loss
            approximator_params['init_type'] = args.init_type

        if args.alg in ['boot', 'particle']:
            approximator_params['n_approximators'] = args.n_approximators
            algorithm_params['n_approximators'] = args.n_approximators

        agent = agent_algorithm(approximator, pi, mdp.info,
                          approximator_params=approximator_params,
                          **algorithm_params)

        if args.ucb:
            q = agent.approximator
            if args.alg == 'particle':
                def mu(state):
                    q_list = q.predict(state).squeeze()
                    qs = np.array(q_list)
                    return qs.mean(axis=0)

                quantiles = [i * 1. / (args.n_approximators - 1) for i in range(args.n_approximators)]
                for p in range(args.n_approximators):
                    if quantiles[p] >= 1 - args.delta:
                        delta_index = p
                        break

                def quantile_func(state):
                    q_list = q.predict(state).squeeze()

                    qs = np.sort(np.array(q_list), axis=0)
                    return qs[delta_index,:]
                print("Setting up ucb policy")
                pi.set_mu(mu)
                pi.set_quantile_func(quantile_func)

            if args.alg == 'gaussian':
                standard_bound = norm.ppf(1 - args.delta, loc=0, scale=1)
                def mu(state):
                    q_and_sigma = q.predict(state).squeeze()
                    means = q_and_sigma[0]
                    return means

                def quantile_func(state):
                    q_and_sigma = q.predict(state).squeeze()
                    means = q_and_sigma[0]
                    sigmas = q_and_sigma[1]
                    return sigmas * standard_bound + means

                print("Setting up ucb policy")
                pi.set_mu(mu)
                pi.set_quantile_func(quantile_func)
        args.count = 100
        if args.plot_qs:
            import matplotlib.pyplot as plt
            colors = ['red','blue','green']
            labels = ['left','nop','right']

            def plot_probs(qs):
                args.count += 1
                if args.count < 1:
                   return
                ax.clear()
                for i in range(qs.shape[-1]):
                    mu = np.mean(qs[...,i],axis=0)
                    sigma = np.std(qs[...,i],axis=0)
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 20)
                    ax.plot(x, stats.norm.pdf(x, mu, sigma),label=labels[i],color=colors[i])

                ax.set_xlabel('Q-value')
                ax.set_ylabel('Probability')
                ax.set_title('Q-distributions')
                #ax.set_ylim(bottom=0, top=1)

                plt.draw()
                plt.pause(0.02)
                #print("Plotted")
                args.count = 0
                #return probs


            plt.ion()
            fig, ax = plt.subplots()

            plot_probs(np.array(agent.approximator.predict(np.array(mdp.reset()))))

            input()
            args.count= 100
            qs = np.array([np.linspace(-1000,0,10), np.linspace(-2000,-1000,10), np.linspace(-750,-250,10)])
            plot_probs(qs.T)
        # Algorithm
        core = Core(agent, mdp)
        core_test = Core(agent, mdp)

        # RUN

        # Fill replay memory with random dataset

        print_epoch(0)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet,
                   )

        if args.save:
            agent.approximator.model.save()

        # Evaluate initial policy
        if hasattr(pi, 'set_eval'):
            pi.set_eval(True)
        pi.set_epsilon(epsilon_test)
        dataset = core_test.evaluate(n_steps=test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        scores.append(get_stats(dataset))
        if args.plot_qs:
            pi.set_plotter(plot_probs)
        np.save(folder_name + '/scores_' + str(ts) + '.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch)
            print('- Learning:')
            # learning step
            if hasattr(pi, 'set_eval'):
                pi.set_eval(False)

            pi.set_epsilon(epsilon)
            # learning step
            if args.plot_qs:
                pi.set_plotter(None)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency,
                       quiet=args.quiet,
                       )

            if args.save:
                agent.approximator.model.save()

            print('- Evaluation:')
            # evaluation step
            if hasattr(pi, 'set_eval'):
                pi.set_eval(True)
            pi.set_epsilon(epsilon_test)
            if args.plot_qs:
                pi.set_plotter(plot_probs)
            dataset = core_test.evaluate(n_steps=test_samples,
                                         render=args.render,
                                         quiet=args.quiet)
            scores.append(get_stats(dataset))
            np.save(folder_name + '/scores_' + str(ts) + '.npy', scores)

    return scores


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_mdp = parser.add_argument_group('Environment')
    arg_mdp.add_argument("--horizon", type=int, default=100)
    arg_mdp.add_argument("--gamma", type=float, default=0.99)
    arg_mdp.add_argument("--name", type=str, default='Gridworld')
    arg_mdp.add_argument("--fast_zone", type=float, default=1.0)
    arg_mdp.add_argument("--slow_zone", type=float, default=10)
    arg_mdp.add_argument("--goal", type=float, default=0.)
    arg_mdp.add_argument('--grid_size', type=int, default=5)
    arg_mdp.add_argument('--rand_initial', action='store_true')

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=100,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=5000,
                         help='Max size of the replay memory.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--n-features", type=int, default=10)
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--net-type",
                         choices=['features',
                                  'linear'],
                         default='features',
                         help='Type of net')
    arg_net.add_argument("--learning-rate", type=float, default=.001,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered')
    arg_net.add_argument("--epsilon", type=float, default=.01,
                         help='Epsilon term used in rmspropcentered')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--alg",
                         choices=['boot',
                                  'particle',
                                  'gaussian',
                                  'dqn'],
                         default='particle',
                         help='Algorithm to use')
    arg_alg.add_argument("--weighted", action='store_true')
    arg_alg.add_argument("--clip-target", action='store_true')
    arg_alg.add_argument("--ucb", action='store_true')
    arg_alg.add_argument("--boot", action='store_true',
                         help="Flag to use BootstrappedDQN.")
    arg_alg.add_argument("--gaussian", action='store_true',
                         help="Flag to use GaussianDQN.")
    arg_alg.add_argument("--double", action='store_true',
                         help="Flag to use the DoubleDQN version of the algorithm.")
    arg_alg.add_argument("--multiple_nets", action='store_true',
                         help="")
    arg_alg.add_argument("--update-type",
                         choices=['mean', 'weighted', 'optimistic'],
                         default='mean',
                         help='Kind of update to perform (only WQL algorithms).')
    arg_alg.add_argument("--n-approximators", type=int, default=10,
                         help="Number of approximators used in the ensemble for"
                              "Averaged DQN.")
    arg_alg.add_argument("--loss",
                         choices=['squared_loss',
                                  'huber_loss',
                                  'triple_loss'
                                  ],
                         default='huber_loss',
                         help="Loss functions used in the approximator")
    arg_alg.add_argument("--delta", type=float, default=0.1,
                         help='Parameter of ucb policy')
    arg_alg.add_argument("--q-max", type=float, default=100,
                         help='Upper bound for initializing the heads of the network')
    arg_alg.add_argument("--q-min", type=float, default=0,
                         help='Lower bound for initializing the heads of the network')
    arg_alg.add_argument("--sigma-weight", type=float, default=1.0,
                         help='Used in gaussian learning to explore more')
    arg_alg.add_argument("--init-type", choices=['boot', 'linspace'], default='linspace',
                         help='Type of initialization for the network')
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=1000,
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
    arg_alg.add_argument("--final-exploration-frame", type=int, default=30000,
                         help='Number of steps until the exploration rate stops'
                              'decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=0.,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=1000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--p-mask", type=float, default=1.)

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--load-path', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--store_prob', action='store_true',
                           help='Flag specifying whether to save probability of exploration.')
    arg_utils.add_argument('--evaluation', action='store_true',
                           help='Flag specifying whether the model loaded will be evaluated.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')
    arg_utils.add_argument('--plot_qs', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')
    arg_utils.add_argument("--device", type=int, default=0,
                           help='Index of the GPU.')
    arg_utils.add_argument("--n_experiments", type=int, default=1,
                           help='Number of experiments to run')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)


    n_experiments = args.n_experiments
    if args.alg == 'boot':
        from boot_net import SimpleNet
        if args.double:
            agent_algorithm = BootstrappedDoubleDQN
        else:
            agent_algorithm = BootstrappedDQN
    elif args.alg == 'gaussian':
        from gaussian_net import SimpleNet
        agent_algorithm = GaussianDQN
    elif args.alg == 'dqn':
        from dqn_net import SimpleNet
        if args.double:
            agent_algorithm = DoubleDQN
        else:
            agent_algorithm = DQN
    else:

        from net import SimpleNet
        if args.double:
            agent_algorithm = ParticleDoubleDQN
        else:
            agent_algorithm = ParticleDQN
    if args.debug or n_experiments == 1:
        out = [experiment(args, agent_algorithm)]
    else:
        out = Parallel(n_jobs=-1)(
        delayed(experiment)(args, agent_algorithm) for _ in range(n_experiments))

    tf.reset_default_graph()

    #np.save(folder_name + '/scores.npy', out)
