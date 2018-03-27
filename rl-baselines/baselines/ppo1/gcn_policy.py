import tensorflow as tf
import numpy as np
import gym
from baselines.common.distributions import make_pdtype,MultiCatCategoricalPdType,CategoricalPdType
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

def bilinear_multi(emb_1, emb_2, out_dim, name='bilinear'):
    node_dim = emb_1.get_shape()[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [out_dim, node_dim, node_dim])
        emb_1 = tf.tile(tf.reshape(emb_1,[1,emb_1.get_shape()[0],emb_1.get_shape()[1]]),[out_dim,1,1])
        emb_2 = tf.transpose(emb_2)
        emb_2 = tf.tile(tf.reshape(emb_2,[1,emb_2.get_shape()[0],emb_2.get_shape()[1]]),[out_dim,1,1])
        return emb_1 @ W @ emb_2



class GCNPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='small'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        self.pdtype = MultiCatCategoricalPdType

        # fixed size ob
        # ob = {'adj':U.get_placeholder(name="adj", dtype=tf.float32, shape=ob_space['adj'].shape),
        #       'node':U.get_placeholder(name="node", dtype=tf.float32, shape=ob_space['node'].shape)}
        ob = {'adj': U.get_placeholder(name="adj", dtype=tf.float32, shape=[ob_space['adj'].shape[0],None,None]),
              'node': U.get_placeholder(name="node", dtype=tf.float32, shape=[1,None,ob_space['node'].shape[2]])}
        print('ob_adj', ob['adj'].get_shape())
        if kind == 'small': # from A3C paper
            emb_node = GCN(ob['adj'], ob['node'], 32, name='gcn1')
            emb_node = GCN(ob['adj'], emb_node, 32, is_act=False, is_normalize=True, name='gcn2')
            emb_node = tf.reshape(emb_node,shape=[-1,emb_node.get_shape()[-1]])  # n*f
            print('emb_node', emb_node.get_shape())

        else:
            raise NotImplementedError

        # 2 get graph embedding
        emb_graph = tf.reduce_max(emb_node, axis=0)  # max pooling
        # 3.1: select first(active) node
        # rules: do not select isolated nodes
        logits_first = tf.layers.dense(emb_node[:-3,:], 32, activation=tf.nn.relu, name='linear_select1') # todo: feed in parameters for -3 !!
        logits_first = tf.layers.dense(logits_first, 1, activation=None, name='linear_select2')
        pd_first = CategoricalPdType(-1).pdfromflat(flat=tf.transpose(logits_first))
        ac_first = pd_first.sample()
        print('logits_first',logits_first.get_shape())
        print('ac_first',ac_first.get_shape())
        emb_first = tf.gather(params=emb_node,indices=ac_first,axis=0)
        # 3.2: select second node
        # rules: do not select first node
        logits_second = tf.transpose(bilinear(emb_first,emb_node,name='logits_second')) # n*1 todo:eliminate impossible selections
        mask = tf.one_hot(indices=ac_first,depth=tf.shape(emb_node)[0],on_value=0,off_value=1)
        logits_second = tf.boolean_mask(tensor=logits_second,mask=mask,axis=0)
        logits_second = tf.reshape(logits_second,shape=[-1,1])
        print(logits_first.get_shape(),logits_second.get_shape())
        pd_second = CategoricalPdType(-1).pdfromflat(tf.transpose(logits_second))
        ac_second = pd_second.sample()
        ac_second += tf.cast(tf.greater_equal(ac_second,ac_first),dtype=tf.int64)
        emb_second = tf.gather(params=emb_node,indices=ac_second,axis=0)
        # 3.3 predict edge type
        logits_edge = tf.squeeze(bilinear_multi(emb_first,emb_second,out_dim=ob['adj'].get_shape()[0]),squeeze_dims=-1)
        pd_edge = CategoricalPdType(-1).pdfromflat(tf.transpose(logits_edge))
        ac_edge = pd_edge.sample()
        print('logits_edge', logits_edge.get_shape())
        print('ac_edge', ac_edge.get_shape())

        ncat_list = [ob_space['adj'].shape[-1],ob_space['adj'].shape[-1],ob_space['adj'].shape[0]]
        self.pd = self.pdtype(ncat_list).pdfromflat([logits_first,logits_second,logits_edge])
        self.vpred = tf.reduce_max(tf.layers.dense(emb_node, 1, name='value', kernel_initializer=U.normc_initializer(1.0)),axis=0)

        self.state_in = []
        self.state_out = []

        self.ac = tf.concat((ac_first,ac_second,ac_edge),axis=0)

        debug = {}
        debug['logits_first'] = logits_first
        debug['logits_second'] = logits_second

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self._act = U.function([stochastic, ob['adj'], ob['node']], [self.ac, self.vpred]) # add debug in second arg if needed

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
    ob_space['adj'] = gym.Space(shape=[3,5,5])
    ob_space['node'] = gym.Space(shape=[1,5,3])
    policy = GCNPolicy(name='policy',ob_space=ob_space,ac_space=None)
    stochastic = True
    ob = {}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0,100):
            ob['adj'] = np.ones([3, 7, 7])
            ob['node'] = np.ones([1, 7, 3])
            ac,vpred,debug = policy.act(stochastic,ob)
            if ac[0]==ac[1]:
                print('error')
            else:
                print('i',i,'ac',ac,'vpred',vpred,'debug',debug['logits_first'].shape,debug['logits_second'].shape)