from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from trident_cfg import STORY_DATA_PATH, VOCAB_PATH, EMBED_PATH
from util import load_vocab
from os.path import join as pjoin
from data.story_loader import StoryLoader


def create_variable(name, value, shape, dtype=tf.float32, trainable=True):
    """
    dtype and shape must be the exact same as value
    """
    return tf.get_variable(name=name, shape=shape, dtype=dtype, trainable=trainable,
                           initializer=lambda shape, dtype: tf.cast(value, dtype=dtype))


src_steps = 5
source_tokens = tf.placeholder(tf.int32, shape=[None, src_steps], name='srcInput')

loader = StoryLoader(STORY_DATA_PATH,
                     batch_size=50, src_seq_len=65,
                     tgt_seq_len=20, mode='merged')

embedding = loader.get_w2v_embed().astype('float32')


def load_embedding():
    # a test function
    with vs.variable_scope('embedding') as scope:
        embed = tf.get_variable("L_enc", [30, embedding.shape[1]])
        encoder_inputs = embedding_ops.embedding_lookup(embed, source_tokens)
        return encoder_inputs


if __name__ == '__main__':
    encoder_inputs = load_embedding()
    encoder_cell = rnn_cell.GRUCell(256, input_size=embedding.shape[1])
    with vs.variable_scope("Encoder"):
        inp = tf.get_variable('fakeInput',shape=[1, src_steps, embedding.shape[1]])
        out = None
        print tf.shape(encoder_inputs)
        for i in xrange(1):
            with vs.variable_scope("EncoderCell%d" % i) as scope:
                out, _ = rnn.dynamic_rnn(encoder_cell, inp, time_major=False,
                                         dtype=dtypes.float32,
                                         sequence_length=src_steps, scope=scope)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        real_inp = sess.run(encoder_inputs, feed_dict={source_tokens: [[4, 1, 6, 8, 10]]})
        print real_inp.shape
