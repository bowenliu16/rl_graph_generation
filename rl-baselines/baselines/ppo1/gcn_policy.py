import tensorflow as tf
import numpy as np
import gym
from baselines.common.distributions import make_pdtype,TwoCategoricalPdType,CategoricalPdType
import baselines.common.tf_util as U


# gcn mean aggregation over edge features
def GCN(adj, node_feature, out_channels, is_act=True, is_normalize=False, name='gcn_simple'):
    '''
    state s: (adj,node_feature)
    :param adj: b*n*n
    :param node_feature: 1*n*d
    :param out_channels: scalar
    :param name:
    :return:
    '''
    edge_dim = adj.get_shape()[0]
    in_channels = node_feature.get_shape()[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [edge_dim, in_channels, out_channels])
        b = tf.get_variable("b", [edge_dim, 1, out_channels])
        node_embedding = adj@tf.tile(node_feature,[edge_dim,1,1])@W+b
        if is_act:
            node_embedding = tf.nn.relu(node_embedding)
        # todo: try complex aggregation
        node_embedding = tf.reduce_mean(node_embedding,axis=0,keepdims=True) # mean pooling
        if is_normalize:
            node_embedding = tf.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

def bilinear(emb_1, emb_2, name='bilinear'):
    node_dim = emb_1.get_shape()[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [node_dim, node_dim])
        return emb_1 @ W @ tf.transpose(emb_2)

# class GCNPolicy(object):
#     recurrent = False
#     def __init__(self, name, ob_space, ac_space, kind='simple'):
#         with tf.variable_scope(name):
#             self._init(ob_space, ac_space, kind)
#             self.scope = tf.get_variable_scope().name
#
#     def _init(self, ob_space, ac_space, gcn_type, policy_type):
#         self.pdtype = pdtype = make_pdtype(ac_space)
#
#         ob = {}
#         ob['adj'] = tf.placeholder(shape=ob_space['adj'],dtype=tf.float32)
#         ob['node'] = tf.placeholder(shape=ob_space['node'],dtype=tf.float32)
#
#         # 1 get node embedding
#         if gcn_type == 'simple': # from A3C paper
#             emb_node = GCN(ob['adj'],ob['node'],32,name='gcn1')
#             emb_node = GCN(ob['adj'],emb_node,32,is_act=False,is_normalize=True,name='gcn2')
#             emb_node = tf.squeeze(emb_node) # n*f
#         else:
#             raise NotImplementedError
#         # 2 get graph embedding
#         emb_graph = tf.reduce_max(emb_node, axis=0)  # max pooling
#         # 3 get pairwise similarity
#         if policy_type == 'direct':
#             dot_node = emb_node @ tf.transpose(emb_node) # todo: try bilinear
#         else:
#             # first select a node's emb
#             select_score = tf.layers.dense(emb_node, 32, activation=tf.nn.relu, name='linear_select1')
#             select_score = tf.layers.dense(select_score, 1, activation=None,
#                                            name='linear_select2')  # add_edge_score, add_node_score
#             select_first = tf.arg_max(select_score)  # get the argmax score todo: try sample
#             emb_select = emb_node[select_first, :]
#             # prob_select =
#             # then make prediction
#             dot_node = emb_select @ tf.transpose(emb_node) # 1*n
#             select_second = tf.arg_max(dot_node)
#
#         logits = tf.layers.dense(x, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
#         self.pd = pdtype.pdfromflat(logits)
#         self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))[:,0]
#
#         self.state_in = []
#         self.state_out = []
#
#         stochastic = tf.placeholder(dtype=tf.bool, shape=())
#         ac = self.pd.sample() # XXX
#         self._act = U.function([stochastic, ob], [ac, self.vpred])
#
#     def act(self, stochastic, ob):
#         ac1, vpred1 =  self._act(stochastic, ob)
#         return ac1[0], vpred1[0]
#     def get_variables(self):
#         return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
#     def get_trainable_variables(self):
#         return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
#     def get_initial_state(self):
#         return []




class GCNPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='small'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        # assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = TwoCategoricalPdType
        sequence_length = None

        ob = {'adj':U.get_placeholder(name="adj", dtype=tf.float32, shape=ob_space['adj'].shape),
              'node':U.get_placeholder(name="node", dtype=tf.float32, shape=ob_space['node'].shape)}
        if kind == 'small': # from A3C paper
            emb_node = GCN(ob['adj'], ob['node'], 32, name='gcn1')
            emb_node = GCN(ob['adj'], emb_node, 32, is_act=False, is_normalize=True, name='gcn2')
            emb_node = tf.squeeze(emb_node)  # n*f
        else:
            raise NotImplementedError
        # 2 get graph embedding
        emb_graph = tf.reduce_max(emb_node, axis=0)  # max pooling
        # 3.1: select a node's emb
        logits_first = tf.layers.dense(emb_node, 32, activation=tf.nn.relu, name='linear_select1')
        logits_first = tf.layers.dense(logits_first, 1, activation=None, name='linear_select2')
        pd_first = CategoricalPdType(ob_space['adj'].shape[-1]).pdfromflat(flat=tf.transpose(logits_first))
        ac_first = pd_first.sample()
        print('logits_first',logits_first.get_shape())
        print('ac_first',ac_first.get_shape(),emb_node.get_shape())
        emb_first = tf.gather(params=emb_node,indices=ac_first,axis=0)
        # 3.2: predict link with the selected node
        # then make prediction
        logits_second = tf.transpose(bilinear(emb_first,emb_node))  # 1*n todo:eliminate impossible selections
        print('logits_second')
        pd_second = CategoricalPdType(ob_space['adj'].shape[-1]).pdfromflat(tf.transpose(logits_second))
        ac_second = pd_second.sample()

        print(logits_first.get_shape(),logits_second.get_shape())
        self.pd = TwoCategoricalPdType(ob_space['adj'].shape[-1],ob_space['adj'].shape[-1]).pdfromflat(logits_first,logits_second)
        self.vpred = tf.reduce_max(tf.layers.dense(emb_node, 1, name='value', kernel_initializer=U.normc_initializer(1.0)),axis=0)

        self.state_in = []
        self.state_out = []

        self.ac = tf.concat((ac_first,ac_second),axis=0)

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self._act = U.function([stochastic, ob['adj'], ob['node']], [self.ac, self.vpred])

    def act(self, stochastic, ob):
        return self._act(stochastic, ob['adj'], ob['node'])
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []






if __name__ == "__main__":
    adj_np = np.ones((3,4,4))
    adj = tf.placeholder(shape=(3,4,4),dtype=tf.float32)
    node_feature_np = np.ones((1,4,3))
    node_feature = tf.placeholder(shape=(1,4,3),dtype=tf.float32)

    # x = GCN(adj, node_feature, 10, is_act=True, name='gcn1')
    # y = GCN(adj, x, 2, is_act=False, is_normalize=True, name='gcn2')
    #
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #
    #     sess.run(init)
    #     x_np,y_np = sess.run([x,y],feed_dict={adj:adj_np,node_feature:node_feature_np})
    #     print(x_np)
    #     print(x_np.shape)
    #     print(y_np)
    #     print(y_np.shape)
    #
    #     # print(sess.run(x, feed_dict={adj: adj_np, node_feature: node_feature_np}))
    ob_space = {}
    ob_space['adj'] = gym.Space(shape=[2,5,5])
    ob_space['node'] = gym.Space(shape=[1,5,3])
    policy = GCNPolicy(name='policy',ob_space=ob_space,ac_space=None)
    stochastic = True
    ob = {}
    ob['adj'] = np.ones([2,5,5])
    ob['node'] = np.ones([1,5,3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1):
            ac,vpred = policy.act(stochastic,ob)
            print('ac',ac,'vpred',vpred)