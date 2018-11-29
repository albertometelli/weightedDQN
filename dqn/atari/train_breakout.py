import argparse
import os
import sys
import numpy as np
sys.path.append('..')
sys.path.append('../..')
try:
    sys.path.remove('/home/alberto/baselines')
except:
    print("")
from baselines import weighted_deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from eval_policy import eval_atari

def main():
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()
    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument("--device", type=int, default=3,
                           help='Index of the GPU.')
    arg_utils.add_argument("--mean_update", action='store_true')
    arg_utils.add_argument("--verbose", action='store_true')
    arg_utils.add_argument("--interactive", action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    logger.configure()
    env = make_atari('BreakoutNoFrameskip-v4')
    env = bench.Monitor(env, logger.get_dir())
    env = weighted_deepq.wrap_atari_dqn(env)

    def eval_policy_closure(**args):
        return eval_atari(env, **args)

    model = weighted_deepq.learn(
        env,
        "conv_only",
        lr=0.00025,
        total_timesteps=int(5e7),
        batch_size=32,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        train_freq=4,
        learning_starts=50000,
        eval_freq=250000,
        eval_timesteps=125000,
        verbose=args.verbose,
        interactive=args.interactive,
        eval_policy=eval_policy_closure,
        target_network_update_freq=10000,
        gamma=0.99,
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=False,
        weighted_update=not args.mean_update,
        checkpoint_path="deepq_logs/Breakout/"+("mean_update" if args.mean_update else "weighted_update")
    )

    model.save('breakout_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
