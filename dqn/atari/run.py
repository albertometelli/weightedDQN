import argparse
import os
import sys
import numpy as np
sys.path.append('..')
sys.path.append('../..')

from joblib import Parallel, delayed


"""
This script can be used to run Atari experiments with DQN.

"""

# Disable tf cpp warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def experiment():
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          type=str,
                          default='BreakoutNoFrameskip-v4',
                          help='Gym ID of the Atari game.')
    arg_game.add_argument("--screen-width", type=int, default=84,
                          help='Width of the game screen.')
    arg_game.add_argument("--screen-height", type=int, default=84,
                          help='Height of the game screen.')
    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=50000,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=1000000,
                         help='Max size of the replay memory.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='rmsprop',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate", type=float, default=.00025,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--decay", type=float, default=.95)
    arg_net.add_argument("--epsilon", type=float, default=1e-8)

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--weighted", action='store_true')
    arg_alg.add_argument("--weighted-update", action='store_true')
    arg_alg.add_argument("--n-approximators", type=int, default=10,
                         help="Number of approximators used in the ensemble for"
                              "Averaged DQN.")
    arg_alg.add_argument("--loss", 
                             choices=['squared_loss',
                                      'huber_loss',
                                     ],
                         default='squared_loss',
                         help="Loss functions used in the approximator")
    arg_alg.add_argument("--q-max", type=float, default=300,
                         help='Upper bound for initializing the heads of the network')
    arg_alg.add_argument("--q-min", type=float, default=0,
                         help='Lower bound for initializing the heads of the network')
    arg_alg.add_argument("--init-type", choices=['boot', 'linspace'], default='boot',
                         help='Type of initialization for the network')
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=4,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=10000,
                         help='Number of learning step before each update of'
                              'the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=250000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--train-frequency", type=int, default=4,
                         help='Number of learning steps before each fit of the'
                              'neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=1,
                         help='Number of steps until the exploration rate stops'
                              'decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=0.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=0.,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=.005,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=125000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--max-no-op-actions", type=int, default=30,
                         help='Maximum number of no-op action performed at the'
                              'beginning of the episodes. The minimum number is'
                              'history_length.')

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
    arg_utils.add_argument("--device", type=int, default=0,
                          help='Index of the GPU.')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    from particle_dqn import ParticleDQN
    import tensorflow as tf

    from mushroom.core.core import Core
    from mushroom.environments import Atari
    from mushroom.utils.dataset import compute_scores
    from mushroom.utils.parameters import LinearDecayParameter, Parameter

    from policy import BootPolicy, WeightedPolicy, VPIPolicy
    from net import ConvNet

    def get_stats(dataset):
        score = compute_scores(dataset)
        print('min_reward: %f, max_reward: %f, mean_reward: %f,'
              ' games_completed: %d' % score)

        return score

    scores = list()

    # Evaluation of the model provided by the user.
    if args.load_path:
        mdp = Atari(args.name, args.screen_width, args.screen_height,
                    ends_at_life=False, history_length=args.history_length,
                    max_no_op_actions=args.max_no_op_actions)
        print("Evaluation Run")

        # Policy
        epsilon_test = Parameter(value=args.test_exploration_rate)

        pi = VPIPolicy(args.n_approximators, epsilon=epsilon_test)

        # Approximator
        input_shape = (args.screen_height, args.screen_width,
                       args.history_length)
        approximator_params = dict(
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            n_approximators=args.n_approximators,
            name='test',
            load_path=args.load_path,
            q_min=args.q_min, 
            q_max=args.q_max,
            init_type=args.init_type,
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'decay': args.decay,
                       'epsilon': args.epsilon},
            loss=args.loss
        )

        approximator = ConvNet

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            initial_replay_size=1,
            max_replay_size=1,
            clip_reward=True,
            train_frequency=args.train_frequency,
            n_approximators=args.n_approximators,
            target_update_frequency=args.target_update_frequency,
            weighted_update=args.weighted_update
        )
        agent = ParticleDQN(approximator, pi, mdp.info,
                          approximator_params=approximator_params,
                          **algorithm_params)
        print(agent)
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
        policy_name = 'weighted' if args.weighted else 'vpi'
        update_rule = 'weighted_update' if args.weighted_update else 'max_mean_update'
        # Summary folder
        folder_name = './logs/' + policy_name + '/' +update_rule+'/'+ args.name+"/"+args.loss+"/"+str(args.n_approximators)+"_particles"

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
        mdp = Atari(args.name, args.screen_width, args.screen_height,
                    ends_at_life=True, history_length=args.history_length,
                    max_no_op_actions=args.max_no_op_actions)

        # Policy
        epsilon = LinearDecayParameter(value=args.initial_exploration_rate,
                                       min_value=args.final_exploration_rate,
                                       n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1.)

        if not args.weighted:
            pi = VPIPolicy(args.n_approximators)
        else:
            pi = WeightedPolicy(args.n_approximators)

        # Approximator
        input_shape = (args.screen_height, args.screen_width,
                       args.history_length)
        approximator_params = dict(
            input_shape=input_shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            n_approximators=args.n_approximators,
            folder_name=folder_name,
            q_min=args.q_min, 
            q_max=args.q_max,
            init_type=args.init_type,
            loss=args.loss, 
            optimizer={'name': args.optimizer,
                       'lr': args.learning_rate,
                       'decay': args.decay,
                       'epsilon': args.epsilon}
        )

        approximator = ConvNet

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            clip_reward=True,
            train_frequency=args.train_frequency,
            n_approximators=args.n_approximators,
            target_update_frequency=target_update_frequency,
            weighted_update=args.weighted_update
            )

        agent = ParticleDQN(approximator, pi, mdp.info,
                          approximator_params=approximator_params,
                          **algorithm_params)
        # Algorithm
        core = Core(agent, mdp)

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
        mdp.set_episode_end(False)
        dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                quiet=args.quiet)
        scores.append(get_stats(dataset))

        np.save(folder_name + '/scores.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch)
            print('- Learning:')
            # learning step
            pi.set_eval(False)
            pi.set_epsilon(epsilon)
            mdp.set_episode_end(True)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency, quiet=args.quiet)

            if args.save:
                agent.approximator.model.save()

            print('- Evaluation:')
            # evaluation step
            pi.set_eval(True)
            pi.set_epsilon(epsilon_test)
            mdp.set_episode_end(False)
            dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                    quiet=args.quiet)
            scores.append(get_stats(dataset))

            np.save(folder_name + '/scores.npy', scores)

    return scores


if __name__ == '__main__':
    n_experiments = 1

    out = Parallel(n_jobs=-1)(
        delayed(experiment)() for _ in range(n_experiments))
    tf.reset_default_graph()
