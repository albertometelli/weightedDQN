import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import time
import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import weighted_deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.weighted_deepq.models import build_weighted_q_func, build_q_func



class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = weighted_deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          config=None,
          seed=None,
          lr_q=5e-4,
          lr_sigma=5e-7,
          sigma_weight=0.1,
          total_timesteps=100000,
          buffer_size=50000,
          optimizer="Adam",
          momentum=0.9,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          double_network=False,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          eval_freq=None,
          eval_timesteps=50000,
          weighted_update=True,
          callback=None,
          load_path=None,
          eval_policy=None,
          verbose=False,
          interactive=False,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)



    if double_network:
        build_train = weighted_deepq.build_train_double
        q_func = build_q_func(network, **network_kwargs)
    else:
        build_train = weighted_deepq.build_train
        q_func = build_weighted_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    optimizer_params = {
        'learning_rate': lr_sigma,
    }
    optimizer_dict = {
        "Adam": (tf.train.AdamOptimizer, optimizer_params),
        "Momentum": (tf.train.MomentumOptimizer, {"momentum": momentum, **optimizer_params}),
        "SGD": (tf.train.GradientDescentOptimizer, optimizer_params),
        "RmsProp": (tf.train.RMSPropOptimizer, optimizer_params)
    }

    opt = optimizer_dict[optimizer]
    act, train, update_target, debug = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=opt[0](learning_rate=lr_q),
        sigma_optimizer=opt[0](**opt[1]),
        sigma_weight=sigma_weight,
        gamma=gamma,
        grad_norm_clipping=10,
    )
    train_writer = tf.summary.FileWriter(checkpoint_path,
                                         sess.graph)
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=exploration_final_eps,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    eval_count = 0
    best_rew = -np.inf
    eval_rewards = []
    scores_file = checkpoint_path + '/scores_' + str(time.time())
    first_eval = True
    train_count = 0
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))


        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            q_val, sigma_val, action, samples, eps = act(np.array(obs)[None], update_eps=update_eps, eval_flag=False, **kwargs)
            #train_writer.add_summary(merged, t)

            if verbose:
                print("Q values: {}".format(q_val))
                print("Sigma values: {}".format(sigma_val))
                print("Action: {}".format(action))
                print("Samples: {}".format(samples))
                print("Epsilon: {}".format(eps))

            if interactive:
                input()
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                td_errors, train_qs, train_sigmas, target_qs, target_sigmas, prob, target, summaries = train(obses_t, actions, rewards, obses_tp1, dones, weights, weighted_update)
                if verbose:
                    print("TD-Errors:")
                    print(td_errors)
                    print("Train Qs:")
                    print(train_qs)
                    print("Train Sigmas:")
                    print(train_sigmas)
                    print("Target Qs:")
                    print(target_qs)
                    print("Target sigmas:")
                    print(target_sigmas)
                    print("Prob")
                    print(prob)
                    print("Target")
                    print(target)
                if interactive:
                    input()

                if (np.sum(prob, axis=1) > 1.2).any():
                    print("Prob")
                    print(np.sum(prob, axis=1))
                    input()

                if (np.sum(prob, axis=1) < 0.9).any():
                    print("Prob")
                    print(np.sum(prob, axis=1))
                    input()
                train_writer.add_summary(summaries, train_count)
                train_count += 1
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
                if first_eval:
                    first_eval = False

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

            if  eval_freq is not None and t % eval_freq == 0:
                print("Start eval of {} timesteps, with model after {} steps of training:".format(eval_timesteps, t))
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                checkpoint_name = checkpoint_path + '/checkpoint_eps_' + str(eval_count)
                #save_variables(checkpoint_name)

                def pi_wrapper(ob):
                    a = act(np.array(ob)[None], update_eps=.005, eval_flag=True, **kwargs)[2]

                    return a

                rew_eval, _, = eval_policy(pi=pi_wrapper, n_timesteps=eval_timesteps, verbose=False)
                eval_rewards.append(rew_eval)
                np.save(scores_file, eval_rewards)
                print("Finished eval:   Score:{}".format(rew_eval))
                if rew_eval > best_rew:
                    print("New best model with evaluation saved")
                    checkpoint_name = checkpoint_path + '/best_eval'
                    save_variables(checkpoint_name)
                    best_rew = rew_eval
                eval_count += 1
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
