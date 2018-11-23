from baselines.weighted_deepq import models  # noqa
from baselines.weighted_deepq.build_graph import build_act, build_train  # noqa
from baselines.weighted_deepq.weighted_deepq import learn, load_act  # noqa
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
