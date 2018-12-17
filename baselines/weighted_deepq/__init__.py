from baselines.weighted_deepq import models  # noqa
from baselines.weighted_deepq.build_graph import build_act, build_train, build_train_double, build_train_particle  # noqa
from baselines.weighted_deepq.weighted_deepq import learn, load_act# noqa
from baselines.weighted_deepq.particle_deepq import learn as particle_learn
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env,episode_life=True):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, episode_life=episode_life, frame_stack=True, scale=True)
