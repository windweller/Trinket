from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


def create_variable(name, value, shape, dtype=tf.float32, trainable=True):
    """
    dtype and shape must be the exact same as value
    """
    return tf.get_variable(name=name, shape=shape, dtype=dtype, trainable=trainable,
                           initializer=lambda shape, dtype: tf.cast(value, dtype=dtype))


_VARSTORE_KEY = ("__variable_store",)
_VARSCOPE_KEY = ("__varscope",)

store = ops.get_collection(_VARSTORE_KEY)
print store

with vs.variable_scope('foo') as s:
    # a = tf.Variable(3, trainable=True, name="RandomVar")
    # a = tf.get_variable('RandomVar', shape=[1], trainable=True,
    #                     initializer=lambda shape, dtype: tf.constant(5, dtype=dtype))
    a = create_variable('RandomVar', value=5, shape=[1], trainable=True)

store = ops.get_collection(_VARSTORE_KEY)
print store

with tf.variable_scope("foo") as foo_scope:
    foo_scope.reuse_variables()
    b = tf.get_variable('RandomVar')

# print a == b
#
# print a.name

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print b.eval()

with tf.variable_scope("bar") as bar_scope:
    c = tf.get_variable('barVar', shape=[1])

with tf.variable_scope("bar", reuse=True):
    d = tf.get_variable('barVar', shape=[1])

store = ops.get_collection(_VARSTORE_KEY)
print store

scopes = ops.get_collection(_VARSCOPE_KEY)

print scopes

print ops.get_default_graph()._collections


def load_embedding():
    # a test function
    with vs.variable_scope('embedding') as scope:
        embedding = vs.get_variable('embedding', shape=[100, 50])


with tf.variable_scope("foo", reuse=True) as scope:
    print scope == foo_scope

print tf.trainable_variables()[0].name
