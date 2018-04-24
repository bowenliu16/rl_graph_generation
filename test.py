import tensorflow as tf
import numpy as np
import networkx as nx
from rdkit import Chem
import random

# a = tf.random_normal((5,2))
# b = tf.reduce_max(a,axis=1)
#
# sess = tf.Session()
# print(sess.run(b))
#
# A = tf.random_normal((3,3))
# A = tf.matrix_set_diag(A,tf.ones(A.get_shape(-1),dtype=tf.float32))
# arg = tf.argmax(A)
# dot_a = A @ tf.transpose(A)
# with tf.Session() as sess:
#     print(sess.run(A))
#     print(sess.run(arg))



# a = tf.constant(np.arange(1, 13, dtype=np.int32),
#                 shape=[1, 2, 2, 3])
# b = tf.constant(np.arange(13, 19, dtype=np.int32),
#                 shape=[1, 1, 3, 2])
# b = tf.tile(b,[1,2,1,1])
# c = a@b
# d = a[0,1]@b[0,0]
#
# e = tf.ones([1,3,1,4])
# f = tf.ones([3,1,3,4])
# g = e+f
#
# v = tf.get_variable('v',shape=[1,3,3])
# v_t = tf.tile(v,[2,1,1])
# loss = tf.nn.l2_loss(v_t-tf.zeros(shape=(2,3,3)))
# grad_v = tf.gradients(loss,v)
# grad_vt = tf.gradients(loss,v_t)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     print(sess.run([v_t,grad_v,grad_vt]))


# ob = np.ones([1,5])
# ob_list = ob[None]
# # ob_list = np.array([ob,ob,ob])
# print(ob.shape)
# print(ob_list.shape)

# a = tf.get_variable(name='a',shape=[4,5])
#
#
# a = a[1:4].assign(tf.zeros([3]))
# loss = tf.nn.l2_loss(a-tf.zeros(4))
# grad = tf.gradients(loss,a)


a = tf.get_variable(name='a',shape=[4,3,2])
# b = tf.get_variable(name='b',shape=[4,3,2],trainable=False)
b = tf.ones(shape=[4,3,2])*100

cond = tf.constant(np.random.randint(2,size=[4,3,2]),dtype=tf.bool)
result = tf.where(cond,b,a)
loss = tf.nn.l2_loss(result-tf.ones([4,3,2]))
grad_a = tf.gradients(loss,a)
grad_b = tf.gradients(loss,b)
a = a+grad_a[0]

for i in range(10):
    with tf.Session() as sess:
        if i==0:
            sess.run(tf.global_variables_initializer())
        # print(sess.run(cond))
        # print(sess.run(result))
        print('grad_a',sess.run(grad_a)[0])
        # print('grad_b',sess.run(grad_b)[0])
        print('a',sess.run(a)[0])



# from rdkit import Contrib

# import random
# a = ['a','b',1]
# sample = random.sample(a,k=4)
# print(sample)

# import random
# graph = nx.gnp_random_graph(10,0.5)
# edges = graph.edges()
# edges_sub = random.sample(edges,k=3)
#
# graph_sub = nx.Graph(edges_sub)
# graph_sub = max(nx.connected_component_subgraphs(graph_sub), key=len)
# graph_sub = nx.convert_node_labels_to_integers(graph_sub,label_attribute='old')
# print(graph_sub.number_of_nodes(),graph_sub.number_of_edges())
# print(graph_sub.edges())


# a = ['{},']*5
# str = ''.join(a)
# print(str)
# str_content = str[:-1].format(1,1,1,1,1)
# print(str_content)
#
# info = {}
# info['1'] = 1
# info['2'] = 1
# print(len(info))

# a = list(range(5))
# b = random.sample(a,k=5)
# print(b)