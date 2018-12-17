from baselines.deepq import models  # noqa
from baselines.deepq.build_graph import build_act, build_train  # noqa
from baselines.deepq.deepq import learn, load_act  # noqa
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env, episode_life=True):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, episode_life=episode_life, frame_stack=True, scale=False)
