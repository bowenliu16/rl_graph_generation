from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from tensorboardX import SummaryWriter
from baselines.ppo1.gcn_policy import discriminator,discriminator_net

def traj_segment_generator(args, pi, env, horizon, stochastic,discriminator_func):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    ob_adj = ob['adj']
    ob_node = ob['node']

    cur_ep_ret = 0 # return in current episode
    cur_ep_ret_env = 0
    cur_ep_ret_d = 0
    cur_ep_len = 0 # len of current episode
    cur_ep_len_valid = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_rets_d = []
    ep_rets_env = []
    ep_lens = [] # lengths of ...
    ep_lens_valid = [] # lengths of ...
    ep_rew_final = []



    # Initialize history arrays
    # obs = np.array([ob for _ in range(horizon)])
    ob_adjs = np.array([ob_adj for _ in range(horizon)])
    ob_nodes = np.array([ob_node for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        # print('-------ac-call-----------')
        ac, vpred, debug = pi.act(stochastic, ob)
        # print('ob',ob)
        # print('debug ob_len',debug['ob_len'])
        # print('debug logits_first_mask', debug['logits_first_mask'])
        # print('debug logits_second_mask',debug['logits_second_mask'])
        # print('debug logits_first_mask', debug['logits_first_mask'])
        # print('debug logits_second_mask', debug['logits_second_mask'])
        # print('debug',debug)
        # print('ac',ac)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob_adj" : ob_adjs, "ob_node" : ob_nodes, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_lens_valid" : ep_lens_valid, "ep_final_rew":ep_rew_final,"ep_rets_env" : ep_rets_env,"ep_rets_d" : ep_rets_d}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_lens_valid = []
            ep_rew_final = []
            ep_rets_d = []
            ep_rets_env = []

        i = t % horizon
        # obs[i] = ob
        ob_adjs[i] = ob['adj']
        ob_nodes[i] = ob['node']
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew_env, new, info = env.step(ac)
        rew_d = 0 # default
        if rew_env>0: # if action valid
            cur_ep_len_valid += 1
            # add stepwise discriminator reward
            rew_d = args.gan_ratio * (
            1 - discriminator_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :])[0]) / env.max_atom
        rews[i] = rew_d+rew_env

        cur_ep_ret += rew_d+rew_env
        cur_ep_ret_d += rew_d
        cur_ep_ret_env += rew_env
        cur_ep_len += 1

        if new:
            with open('molecule_gen/'+args.dataset+'_'+args.name+'.csv', 'a') as f:
                str = ''.join(['{},']*(len(info)+2))[:-1]+'\n'
                f.write(str.format(info['smile'],info['reward_valid'],info['reward_qed'],info['reward_sa'],info['reward_logp'],rew_env,rew_d,rew_d+rew_env,info['flag_steric_strain_filter'],info['flag_zinc_molecule_filter'],info['stop']))
            ep_rets.append(cur_ep_ret)
            ep_rets_env.append(cur_ep_ret_env)
            ep_rets_d.append(cur_ep_ret_d)
            ep_lens.append(cur_ep_len)
            ep_lens_valid.append(cur_ep_len_valid)
            ep_rew_final.append(rew_env)
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_len_valid = 0
            cur_ep_ret_d = 0
            cur_ep_ret_env = 0
            ob = env.reset()

        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]





def learn(args,env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        writer=None
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    # ob = U.get_placeholder_cached(name="ob")
    ob = {}
    ob['adj'] = U.get_placeholder_cached(name="adj")
    ob['node'] = U.get_placeholder_cached(name="node")

    ob_gen = {}
    ob_gen['adj'] = U.get_placeholder(shape=[None,ob_space['adj'].shape[0],None,None],dtype=tf.float32,name='adj_gen')
    ob_gen['node'] = U.get_placeholder(shape=[None,1,None,ob_space['node'].shape[2]],dtype=tf.float32,name='node_gen')

    # ac = pi.pdtype.sample_placeholder([None])
    # ac = tf.placeholder(dtype=tf.int64,shape=env.action_space.nvec.shape)
    ac = tf.placeholder(dtype=tf.int64, shape=[None,4],name='ac_real')

    ## PPO loss
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    pi_logp = pi.pd.logp(ac)
    oldpi_logp = oldpi.pd.logp(ac)
    ratio_log = pi.pd.logp(ac) - oldpi.pd.logp(ac)

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    ## Expert loss
    loss_expert = -tf.reduce_mean(pi_logp)

    ## Discriminator loss
    loss_discriminator,_,_ = discriminator(ob,ob_gen,name='d_net')
    loss_discriminator_gen = discriminator_net(ob_gen,name='d_net')

    var_list_pi = pi.get_trainable_variables()
    var_list_d = [var for var in tf.global_variables() if 'd_net' in var.name]

    ## debug
    debug={}
    debug['ac'] = ac
    debug['ob_adj'] = ob['adj']
    debug['ob_node'] = ob['node']
    debug['pi_logp'] = pi_logp
    debug['oldpi_logp'] = oldpi_logp
    debug['kloldnew'] = kloldnew
    debug['ent'] = ent
    debug['ratio'] = ratio
    debug['ratio_log'] = ratio_log
    debug['emb_node2'] = pi.emb_node2
    debug['pi_logitfirst'] = pi.logits_first
    debug['pi_logitsecond'] = pi.logits_second
    debug['pi_logitedge'] = pi.logits_edge

    debug['pi_ac'] = pi.ac
    debug['oldpi_logitfirst'] = oldpi.logits_first
    debug['oldpi_logitsecond'] = oldpi.logits_second
    debug['oldpi_logitedge'] = oldpi.logits_edge

    debug['oldpi_ac'] = oldpi.ac

    with tf.variable_scope('pi/gcn1', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('W')
        debug['w'] = w


    ## loss update function
    lossandgrad_ppo = U.function([ob['adj'], ob['node'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list_pi)])
    lossandgrad_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi)])
    lossandgrad_discriminator = U.function([ob['adj'],ob['node'],ob_gen['adj'],ob_gen['node']], [loss_discriminator, U.flatgrad(loss_discriminator, var_list_d)])
    loss_discriminator_gen_func = U.function([ob_gen['adj'],ob_gen['node']], loss_discriminator_gen)



    adam_pi = MpiAdam(var_list_pi, epsilon=adam_epsilon)
    adam_discriminator = MpiAdam(var_list_d, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    #
    # compute_losses_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real],
    #                                 loss_expert)
    compute_losses = U.function([ob['adj'], ob['node'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses)

    U.initialize()
    adam_pi.sync()
    adam_discriminator.sync()


    # Prepare for rollouts
    # ----------------------------------------
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    lenbuffer_valid = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_env = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final = deque(maxlen=100) # rolling buffer for episode rewards


    seg_gen = traj_segment_generator(args, pi, env, timesteps_per_actorbatch, True,loss_discriminator_gen_func)


    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    ## start training
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # logger.log("********** Iteration %i ************"%iters_so_far)
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open('molecule_gen/' + args.dataset+'_'+args.name + '.csv', 'a') as f:
                f.write('***** Iteration {} *****\n'.format(iters_so_far))

        ## Expert
        loss_expert=0
        g_expert=0
        if args.has_expert==1:
            ## Expert train
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for _ in range(optim_epochs*2):
                ob_expert, ac_expert = env.get_expert(optim_batchsize)
                losses_expert, g_expert = lossandgrad_expert(ob_expert['adj'], ob_expert['node'], ac_expert, ac_expert)
                adam_pi.update(g_expert, optim_stepsize * cur_lrmult)
                losses.append(losses_expert)
            loss_expert = np.mean(losses, axis=0, keepdims=True)
            # logger.log(fmt_row(13, loss_expert))


        ## PPO
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob_adj, ob_node, ac, atarg, tdlamret = seg["ob_adj"], seg["ob_node"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob_adj=ob_adj, ob_node=ob_node, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob_adj.shape[0]


        loss_discriminator=0
        g_ppo=0
        g_discriminator=0
        if args.has_rl==1:
            ## PPO train
            assign_old_eq_new() # set old parameter values to new parameter values
            # logger.log("Optimizing...")
            # logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses_ppo = [] # list of tuples, each of which gives the loss for a minibatch
                losses_d = []
                for batch in d.iterate_once(optim_batchsize):
                    # ppo
                    *newlosses, g_ppo = lossandgrad_ppo(batch["ob_adj"], batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam_pi.update(g_ppo, optim_stepsize * cur_lrmult)
                    losses_ppo.append(newlosses)
                    # update discriminator
                    ob_expert, _ = env.get_expert(optim_batchsize)
                    losses_discriminator, g_discriminator = lossandgrad_discriminator(ob_expert["adj"],ob_expert["node"],batch["ob_adj"], batch["ob_node"])
                    adam_discriminator.update(g_discriminator, optim_stepsize * cur_lrmult)
                    losses_d.append(losses_discriminator)
                loss_discriminator = np.mean(losses_d, axis=0, keepdims=True)
                # logger.log(fmt_row(13, np.mean(losses, axis=0)))


        ## PPO val
        # logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob_adj"],batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        # logger.log(fmt_row(13, meanlosses))

        # logger.record_tabular("loss_expert", loss_expert)
        # logger.record_tabular('grad_expert_min',np.amin(g_expert))
        # logger.record_tabular('grad_expert_max',np.amax(g_expert))
        # logger.record_tabular('grad_expert_norm', np.linalg.norm(g_expert))
        # logger.record_tabular('grad_rl_min', np.amin(g))
        # logger.record_tabular('grad_rl_max', np.amax(g))
        # logger.record_tabular('grad_rl_norm', np.linalg.norm(g))
        # logger.record_tabular('learning_rate', optim_stepsize * cur_lrmult)

        if writer is not None:
            writer.add_scalar("loss_expert", loss_expert, iters_so_far)
            writer.add_scalar("loss_discriminator", loss_discriminator, iters_so_far)
            writer.add_scalar('grad_expert_min', np.amin(g_expert), iters_so_far)
            writer.add_scalar('grad_expert_max', np.amax(g_expert), iters_so_far)
            writer.add_scalar('grad_expert_norm', np.linalg.norm(g_expert), iters_so_far)
            writer.add_scalar('grad_rl_min', np.amin(g_ppo), iters_so_far)
            writer.add_scalar('grad_rl_max', np.amax(g_ppo), iters_so_far)
            writer.add_scalar('grad_rl_norm', np.linalg.norm(g_ppo), iters_so_far)
            writer.add_scalar('grad_discriminator_min', np.amin(g_discriminator), iters_so_far)
            writer.add_scalar('grad_discriminator_max', np.amax(g_discriminator), iters_so_far)
            writer.add_scalar('grad_discriminator_norm', np.linalg.norm(g_discriminator), iters_so_far)
            writer.add_scalar('learning_rate', optim_stepsize * cur_lrmult, iters_so_far)

        for (lossval, name) in zipsame(meanlosses, loss_names):
            # logger.record_tabular("loss_"+name, lossval)
            if writer is not None:
                writer.add_scalar("loss_"+name, lossval, iters_so_far)
        # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        if writer is not None:
            writer.add_scalar("ev_tdlam_before", explained_variance(vpredbefore, tdlamret), iters_so_far)
        lrlocal = (seg["ep_lens"],seg["ep_lens_valid"], seg["ep_rets"],seg["ep_rets_env"],seg["ep_rets_d"],seg["ep_final_rew"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens,lens_valid,rews,rews_env,rews_d,rews_final = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        lenbuffer_valid.extend(lens_valid)
        rewbuffer.extend(rews)
        rewbuffer_d.extend(rews_d)
        rewbuffer_env.extend(rews_env)
        rewbuffer_final.extend(rews_final)
        # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        # logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        # logger.record_tabular("EpThisIter", len(lens))
        if writer is not None:
            writer.add_scalar("EpLenMean", np.mean(lenbuffer),iters_so_far)
            writer.add_scalar("EpLenValidMean", np.mean(lenbuffer_valid),iters_so_far)
            writer.add_scalar("EpRewMean", np.mean(rewbuffer),iters_so_far)
            writer.add_scalar("EpRewDMean", np.mean(rewbuffer_d),iters_so_far)
            writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env),iters_so_far)
            writer.add_scalar("EpRewFinalMean", np.mean(rewbuffer_final),iters_so_far)
            writer.add_scalar("EpThisIter", len(lens), iters_so_far)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        # logger.record_tabular("EpisodesSoFar", episodes_so_far)
        # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        # logger.record_tabular("TimeElapsed", time.time() - tstart)
        if writer is not None:
            writer.add_scalar("EpisodesSoFar", episodes_so_far, iters_so_far)
            writer.add_scalar("TimestepsSoFar", timesteps_so_far, iters_so_far)
            writer.add_scalar("TimeElapsed", time.time() - tstart, iters_so_far)
        iters_so_far += 1
        # if MPI.COMM_WORLD.Get_rank()==0:
        #     logger.dump_tabular()




def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
