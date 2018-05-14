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
from gym_molecule.envs.molecule import GraphEnv

def train(args,seed,writer=None):
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
    if args.env=='molecule':
        env = gym.make('molecule-v0')
        env.init(data_type=args.dataset,logp_ratio=args.logp_ratio,qed_ratio=args.qed_ratio,sa_ratio=args.sa_ratio,reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,reward_type=args.reward_type,reward_target=args.reward_target,has_feature=bool(args.has_feature)) # remember call this after gym.make!!
    elif args.env=='graph':
        env = GraphEnv()
        env.init(reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,dataset=args.dataset) # remember call this after gym.make!!
    print(env.observation_space)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        # return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
        return gcn_policy.GCNPolicy(name=name, ob_space=ob_space, ac_space=ac_space, atom_type_num=env.atom_type_num,args=args)
    # env = bench.Monitor(env, logger.get_dir() and
    #     osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    # env = wrap_deepmind(env)
    # env.seed(workerseed)

    pposgd_simple_gcn.learn(args,env, policy_fn,
        max_timesteps=args.num_steps,
        timesteps_per_actorbatch=256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=args.lr, optim_batchsize=32,
        gamma=1, lam=0.95,
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

def molecule_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', type=str, help='environment name: molecule; graph',
                        default='molecule')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=int(5e7))
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--name_load', type=str, default='0new_concatno_mean_layer3_expert1500')
    # parser.add_argument('--name_load', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='zinc',help='caveman; grid; ba; zinc; gdb')
    parser.add_argument('--dataset_load', type=str, default='zinc')
    parser.add_argument('--reward_type', type=str, default='qed_target',help='logppen;logp_target;qed;qedsa;qed_target;mw_target;gan')
    parser.add_argument('--reward_target', type=float, default=0.5,help='target reward value')
    parser.add_argument('--logp_ratio', type=float, default=1)
    parser.add_argument('--qed_ratio', type=float, default=1)
    parser.add_argument('--sa_ratio', type=float, default=1)
    parser.add_argument('--gan_step_ratio', type=float, default=2)
    parser.add_argument('--gan_final_ratio', type=float, default=2)
    parser.add_argument('--reward_step_total', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)
    # parser.add_argument('--has_rl', type=int, default=1)
    # parser.add_argument('--has_expert', type=int, default=1)
    parser.add_argument('--has_d_step', type=int, default=1)
    parser.add_argument('--has_d_final', type=int, default=1)
    parser.add_argument('--has_ppo', type=int, default=1)
    parser.add_argument('--rl_start', type=int, default=250)
    parser.add_argument('--rl_end', type=int, default=int(1e6))
    parser.add_argument('--expert_start', type=int, default=0)
    parser.add_argument('--expert_end', type=int, default=int(1e6))
    parser.add_argument('--save_every', type=int, default=250)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=250)
    # parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--curriculum', type=int, default=0)
    parser.add_argument('--curriculum_num', type=int, default=6)
    parser.add_argument('--curriculum_step', type=int, default=200)
    parser.add_argument('--supervise_time', type=int, default=4)
    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--graph_emb', type=int, default=0)
    parser.add_argument('--stop_shift', type=int, default=-1)
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_concat', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--gcn_aggregate', type=str, default='mean')# sum, mean, concat
    parser.add_argument('--gan_type', type=str, default='normal')# normal, recommend, wgan
    parser.add_argument('--gate_sum_d', type=int, default=0)
    parser.add_argument('--mask_null', type=int, default=0)
    parser.add_argument('--bn', type=int, default=1)
    parser.add_argument('--name_full',type=str,default='')
    parser.add_argument('--name_full_load',type=str,default='')






    return parser

def main():
    args = molecule_arg_parser().parse_args()
    print(args)
    args.name_full = args.env + '_' + args.dataset + '_' + args.name
    args.name_full_load = args.env + '_' + args.dataset_load + '_' + args.name_load + '_' + str(args.load_step)
    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
    # # new
    # with open('molecule_gen/' + args.name + '.csv', 'a') as f:
    #     f.write('{},{},{},{},{},{},{},{}\n'.format('smile', 'reward_qed', 'reward_logp','reward_sa', 'reward_sum', 'qed_ratio',
    #                                                'logp_ratio', 'sa_ratio'))

    # only keep first worker result in tensorboard
    if MPI.COMM_WORLD.Get_rank() == 0:
        writer = SummaryWriter(comment='_'+args.dataset+'_'+args.name)
    else:
        writer = None
    # try:
    train(args,seed=args.seed,writer=writer)
    # except:
    #     writer.export_scalars_to_json("./all_scalars.json")
    #     writer.close()
    #     pass

if __name__ == '__main__':
    main()
