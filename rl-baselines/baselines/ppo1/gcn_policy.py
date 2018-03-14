import tensorflow as tf
import numpy as np
import gym

# gcn mean aggregation over edge features
def gcn_simple(adj,node_feature,out_channels,is_normalize=False,name='gcn_simple'):
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
    with tf.variable_scope(name):
        W = tf.get_variable("W", [edge_dim, in_channels, out_channels])
        b = tf.get_variable("b", [edge_dim, 1, out_channels])
        node_embedding = adj@tf.tile(node_feature,[edge_dim,1,1])@W+b
        node_embedding = tf.reduce_mean(node_embedding,axis=0,keep_dims=True)# todo(jiaxuan): try complex aggregation
        if is_normalize:
            node_embedding = tf.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

if __name__ == "__main__":
    adj_np = np.ones((3,4,4))
    adj = tf.placeholder(shape=(3,4,4),dtype=tf.float32)
    node_feature_np = np.ones((1,4,3))
    node_feature = tf.placeholder(shape=(1,4,3),dtype=tf.float32)

    x = gcn_simple(adj, node_feature, 10, name='gcn1')
    y = gcn_simple(adj, x, 2, is_normalize=True, name='gcn2')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        x_np,y_np = sess.run([x,y],feed_dict={adj:adj_np,node_feature:node_feature_np})
        print(x_np)
        print(x_np.shape)
        print(y_np)
        print(y_np.shape)

        # print(sess.run(x, feed_dict={adj: adj_np, node_feature: node_feature_np}))
