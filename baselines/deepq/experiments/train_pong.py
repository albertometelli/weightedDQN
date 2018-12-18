import sys
import argparse
import os
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
try:
    sys.path.remove('/home/alberto/baselines')
except:
    print("")
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from eval_policy import eval_atari


def main():
    parser = argparse.ArgumentParser()
    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument("--device", type=int, default=3,
                           help='Index of the GPU.')
    arg_utils.add_argument("--grad_norm", action='store_true')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    logger.configure()
    env = make_atari('BreakoutNoFrameskip-v4')
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    e = make_atari('BreakoutNoFrameskip-v4')
    eval_env = deepq.wrap_atari_dqn(e, episode_life=False)

    def eval_policy_closure(**args):
        return eval_atari(eval_env, **args)
    
    model = deepq.learn(
        env,
        "conv_only",
        lr=0.00025,
        total_timesteps=int(2e8),
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=100000,
        target_network_update_freq=2000,
        eval_freq=250000,
        eval_timesteps=65000,
        eval_policy=eval_policy_closure,
        checkpoint_path="deepq_logs/Breakout",
        gamma=0.99,
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        grad_norm_clipping=args.grad_norm
    )

    model.save('pong_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
