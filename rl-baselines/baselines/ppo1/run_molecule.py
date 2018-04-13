#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
# from baselines import bench
# import os.path as osp
from baselines import logger
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# from baselines.common.cmd_util import atari_arg_parser
from tensorboardX import SummaryWriter
import os

import gym
# import gym_molecule

def train(args,env_id, num_timesteps, seed,writer=None):
    from baselines.ppo1 import pposgd_simple_gcn, gcn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make('molecule-v0')
    print(env.observation_space)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        # return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
        return gcn_policy.GCNPolicy(name=name, ob_space=ob_space, ac_space=ac_space, atom_type_num=env.atom_type_num)
    # env = bench.Monitor(env, logger.get_dir() and
    #     osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    # env = wrap_deepmind(env)
    # env.seed(workerseed)

    pposgd_simple_gcn.learn(args,env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=64,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=32,
        gamma=0.99, lam=0.95,
        schedule='linear', writer=writer
    )
    env.close()

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID',
                        default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('--name', type=str, default='test')
    return parser

def main():
    args = atari_arg_parser().parse_args()
    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    # new
    with open('molecule_gen/' + args.name + '.csv', 'a') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('smile', 'reward_qed', 'reward_logp','reward_sa', 'reward_sum', 'qed_ratio',
                                                   'logp_ratio', 'sa_ratio'))

    # only keep first worker result in tensorboard
    if MPI.COMM_WORLD.Get_rank() == 0:
        writer = SummaryWriter(comment='_'+args.name)
    else:
        writer = None
    try:
        train(args,args.env, num_timesteps=args.num_timesteps, seed=args.seed,writer=writer)
    except:
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
        pass

if __name__ == '__main__':
    main()
