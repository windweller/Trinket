"""
A seq2seq model for
story training
"""
import argparse
import math
import os
import random
import sys
import time
import random

from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
from tensorflow.python.ops import embedding_ops, array_ops, math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops

from tensorflow.python.ops.seq2seq import attention_decoder
from data.story_loader import StoryLoader
from os.path import join as pjoin

import numpy as np

Unit = {'gru': tf.nn.rnn_cell.GRUCell,
        'lstm': tf.nn.rnn_cell.BasicLSTMCell}


def create_variable(name, value, shape, dtype=tf.float32, trainable=True):
    """
    dtype and shape must be the exact same as value
    """
    return tf.get_variable(name=name, shape=shape, dtype=dtype, trainable=trainable,
                           initializer=lambda shape, dtype: value)


def attention_decoder_with_embedding(decoder_inputs, initial_state, attention_states,
                                     cell, embedding, num_heads=1,
                                     output_size=None, dtype=dtypes.float32, scope=None,
                                     initial_state_attention=False):
    """
    We are not using output_projection because we are NOT using a sampled softmax

    Parameters
    ----------
    decoder_inputs
    initial_state
    attention_states
    cell
    embedding: outside embedding passed in
    num_heads
    output_size
    dtype
    scope
    initial_state_attention

    Returns
    -------

    """
    if output_size is None:
        output_size = cell.output_size

    with vs.variable_scope(scope or "attention_decoder_with_embedding"):
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(
            emb_inp, initial_state, attention_states, cell, output_size=output_size,
            num_heads=num_heads, loop_function=None,
            initial_state_attention=initial_state_attention)


def attention_encoder(decoder_inputs, initial_state, attention_states,
                      cell, num_heads=1,
                      output_size=None, dtype=dtypes.float32, scope=None,
                      initial_state_attention=False):
    """
    Encoder that receives attention from another encoder

    Parameters
    ----------
    decoder_inputs:
        second encoder's input we call it a decoder's input
        it should be already wrapped by add_embedding()
        it's A list of num_steps length 2D Tensors [batch_size, input_size = embed_size]
    initial_state:
        2D Tensor (batch_size x cell.state_size).
    attention_states:
        3D Tensor (batch_size x attn_length (seq_length) x attn_size)
    cell
    num_heads
    output_size
    dtype
    scope
    initial_state_attention

    Returns
    -------
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape (batch_size x cell.state_size).

    """
    decoder_inputs = [decoder_inputs]  # in original model this is a bucket list of inputs

    with vs.variable_scope(scope or "attention_encoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    hidden = array_ops.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    for a in xrange(num_heads):
        k = vs.get_variable("AttnW_%d" % a,
                            [1, 1, attn_size, attention_vec_size])
        hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
        v.append(vs.get_variable("AttnV_%d" % a, [attention_vec_size]))

    def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        ds = []  # Results of attention reads will be stored here.
        for a in xrange(num_heads):
            with vs.variable_scope("Attention_%d" % a):
                y = rnn_cell.linear(query, attention_vec_size, True)
                y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                # Attention mask is a softmax of v^T * tanh(...).
                s = math_ops.reduce_sum(
                    v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                a = tf.nn.softmax(s)
                # Now calculate the attention-weighted vector d.
                d = math_ops.reduce_sum(
                    array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                    [1, 2])
                ds.append(array_ops.reshape(d, [-1, attn_size]))
        return ds

    outputs = []
    batch_attn_size = array_ops.pack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)]

    for a in attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, attn_size])
    if initial_state_attention:
        attns = attention(initial_state)

    state = initial_state

    # this is now iterating on time steps
    for i, inp in enumerate(decoder_inputs):
        if i > 0:
            vs.get_variable_scope().reuse_variables()
        # Merge input and previous attentions into one vector of the right size.
        x = rnn_cell.linear([inp] + attns, cell.input_size, True)
        # Run the RNN.
        cell_output, state = cell(x, state)
        # Run the attention mechanism.
        if i == 0 and initial_state_attention:
            with vs.variable_scope(vs.get_variable_scope(), reuse=True):
                attns = attention(state)
        else:
            attns = attention(state)

        with vs.variable_scope("AttnOutputProjection"):
            output = rnn_cell.linear([cell_output] + attns, output_size, True)

        outputs.append(output)

    # we only want the last state
    return outputs, state


def add_word_embedding(input, time_step, embedding):
    """
    Similar to rnn_cell.EmbeddingWrapper
    but this mainly wraps on an existing embedding

    Parameters
    ----------
    inputs: [batch_size, max_time] shape Tensor/Numpy/Placeholer!
    embedding

    Returns
    embedded_inputs: a list length of batch_size of Tensor [max_time, embed_size]
    -------
    """
    embed = tf.nn.embedding_lookup(embedding, input)
    l = tf.split(1, time_step, embed)
    embed = [tf.squeeze(i, [1]) for i in l]
    return embed


# def attention_seq2seq_with_embedding(encoder_inputs, decoder_inputs, embedding,
#                                      cell, args, num_heads=1, output_projection=None,
#                                      dtype=dtypes.float32,
#                                      scope=None, initial_state_attention=False):
#     """
#     Our twist on "embedding_attention_seq2seq"
#
#     but with pre-initialized embeddings, the
#     rest are the same.
#
#     This function is strictly for pre-training
#
#     Parameters
#     ----------
#     args: pass in args to have a handle on configurations
#     encoder_inputs: 2D int32 Tensors of shape [batch_size, time_steps].
#     decoder_inputs: A place holder of 2D int32 Tensors of shape [batch_size, time_steps].
#     cell: rnn_cell.RNNCell defining the cell function and size.
#
#     Returns
#     -------
#     """
#     with vs.variable_scope(scope or "embedding_attention_seq2seq"):
#         # Encoder.
#
#         # Tensor shape of [batch_size, max_time, cell.input_size]
#         em_encoder_input = add_word_embedding(encoder_inputs, args.time_steps, embedding)
#
#         encoder_outputs, encoder_state = rnn.rnn(
#             cell, em_encoder_input, dtype=dtype)
#
#         # First calculate a concatenation of encoder outputs to put attention on.
#         top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
#                       for e in encoder_outputs]
#         attention_states = array_ops.concat(1, top_states)
#
#         # Decoder.
#         output_size = None
#         if output_projection is None:
#             cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
#             output_size = num_decoder_symbols
#
#         # We don't consider feed_previous being a Tensor,
#         # check original implementation for feed_previous being tensor case
#         return attention_decoder_with_embedding(
#             decoder_inputs, encoder_state, attention_states, cell,
#             num_decoder_symbols, num_heads=num_heads,
#             output_size=output_size,
#             initial_state_attention=initial_state_attention)


def attention_enc2enc_with_embedding(encoder_inputs, decoder_inputs, embedding,
                                     cell, args, num_heads=1,
                                     dtype=dtypes.float32, scope=None):
    """

    We initialize all attentions as zeroes

    Parameters
    ----------
    encoder_inputs
    decoder_inputs
    embedding
    cell
    args
    initial_state:
        2D Tensor (batch_size x cell.state_size)
    num_heads:
        Number of attention heads that read from attention_states.
        We want this to be covering all embedding states (instead of just 1)
    dtype
    scope
    initial_state_attention

    Returns
    -------

    """
    with vs.variable_scope(scope or "attention_enc2enc_with_embedding"):
        # Tensor shape of [batch_size, max_time, cell.input_size = embedding_size]
        em_encoder_input = add_word_embedding(encoder_inputs, args.time_steps, embedding)

        encoder_outputs, encoder_state = rnn.rnn(cell, em_encoder_input, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

    return attention_encoder(decoder_inputs, encoder_state, attention_states,
                             cell, num_heads, dtype=dtypes.float32, scope=None,
                             initial_state_attention=False)


class StorySeq2SeqModel(object):
    """
    An adapatation of seq2seq model of tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py
    Diff:

    We implement bi-rnn
    We also don't do buckets (since it's hard to compare)
    """

    def __init__(self, embedding, args):
        """

        Parameters
        ----------
        args: object (could be FLAGS)

            batch_size: int
                How many are in a batch
            time_steps: int
                Number of time steps (max sequence length)
            rnn_dim: int
                number of units in each layer of the model. (same as 'size' in seq2seq)
            vocab_size: int
                number of vocab in the corpus (maybe split it to target_vocab_size
                and src_vocab_size is better?)
            rlayers: int
                number of layers in the model.
            num_heads: int
                number of encoder states we want attention to work on (should be ALL)
            max_norm: int
                gradients will be clipped to maximally this norm. (set to 5.0 in translate.py)
            lr: float
            lr_decay: float
                decay learning rate by this much when needed.
            unit: {'lstm', 'gru'}
                We can consider adding bi-lstm/bi-gru (bi-rnn) in

            forward_only: boolean
                if set, we do not construct the backward pass in the model. (no learning)
        """
        self.args = args
        self.learning_rate = tf.Variable(float(args.lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * args.lr_decay)
        self.embedding = embedding
        self.label_size = 2
        self.hidden_dim = self.args.rnn_dim

        if self.args.num_heads == 0:
            # paying attention to all encoder states
            self.args.num_heads = self.args.time_steps

        # create placeholders
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.args.time_steps],
                                             name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.args.time_steps],
                                             name='decoder_inputs')
        self.y_labels = tf.placeholder(dtype=tf.int32, shape=[None, self.label_size], name='y_labels')

        # multilayer
        cell = Unit[args.unit](self.hidden_dim)
        if args.rlayers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.rlayers)

        # calculate loss and output
        with vs.variable_scope(vs.get_variable_scope()):
            _, state = attention_enc2enc_with_embedding(self.encoder_inputs, self.decoder_inputs,
                                                        embedding, cell, args, num_heads=args.num_heads,
                                                        dtype=dtypes.float32)

            # state: 2D Tensor of shape (batch_size x cell.state_size)
            o = self.output_projection(state)
            predictions = tf.nn.softmax(o)
            one_hot_prediction = tf.argmax(self.y_labels, 1)
            correct_prediction = tf.equal(
                tf.argmax(self.y_labels, 1), one_hot_prediction)
            self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predictions, self.y_labels)
            self.loss = tf.reduce_mean(cross_entropy)

        self.cost = tf.reduce_mean(cross_entropy) / self.args.batch_size

        if self.args.forward_only:
            return

        self.lr = tf.Variable(self.args.lr, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          self.args.max_norm)
        # sgd
        optimizer = None
        if self.args.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.args.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        """
        Used to update learning rate (for learning rate decay)
        """
        session.run(tf.assign(self.lr, lr_value))

    def run_epoch(self, session, loader,
                  shuffle=False, verbose=True):
        """

        Parameters
        ----------
        session
        loader:
            In this case it should be the story_loader
        shuffle
        verbose

        Returns
        -------

        """

        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        exp_cost = 0.

        total_steps = loader.train_num_batches

        it = 0
        for k in xrange(total_steps):
            tic = time.time()
            it += 1

            x, (y, y_2), label = loader.get_batch('train', k)
            # we are only using y right now. TODO: how to incorporate y_2

            feed = {self.encoder_inputs: x, self.decoder_inputs: y,
                    self.y_labels: tf.one_hot(label, self.label_size, on_value=1, off_value=0)}

            # this part should work
            loss, total_correct, _ = session.run(
                [self.cost, self.correct_predictions, self.train_op],
                feed_dict=feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_loss.append(loss)

            exp_cost += loss

            if verbose and it % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    k, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

    def output_projection(self, h):
        """
        Parameters
        ----------
        h: a hidden state tensor (batch_size x cell.state_size)

        Returns
        -------
        """
        with tf.variable_scope('Softmax'):
            self.U = tf.get_variable('U', shape=[self.hidden_dim,
                                                 self.label_size])
            self.b2 = tf.get_variable('b2', shape=[self.label_size],
                                      initializer=tf.zeros_initializer)
            output = tf.matmul(h, self.U) + self.b2
            return output


def to_parent(path):
    return os.path.abspath(pjoin(path, os.pardir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_dim', type=int, default=256, help='dimension of recurrent states')
    parser.add_argument('--rlayers', type=int, default=2, help='number of hidden layers for RNNs')
    parser.add_argument('--unroll', type=int, default=35, help='number of time steps to unroll for source')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batches')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    # parser.add_argument('--lr_decay_after', type=int, default=10, help='epoch after which to decay')
    parser.add_argument('--lr_decay_threshold', type=float, default=0.01,
                        help='begin decaying learning rate if diff of prev 2 validation costs less than thresohld')
    parser.add_argument('--max_lr_decays', type=int, default=8, help='maximum number of times to decay learning rate')
    # parser.add_argument('--dropout', type=float, default=0.0,
    #                     help='dropout (fraction of units randomly dropped on non-recurrent connections)')
    # parser.add_argument('--recdrop', action='store_true',
    #                     help='use dropout on recurrent updates if True, use stocdrop if False')
    parser.add_argument('--bidir', action='store_true', help='whether to use bidirectional (GRU)')
    parser.add_argument('--max_epochs', type=int, default=400, help='maximum number of epochs to train')
    parser.add_argument('--max_norm', type=float, default=5.0, help='gradient clipping in 2-norm')
    parser.add_argument('--unit', type=str, choices=['gru', 'lstm'], default='lstm')
    parser.add_argument('--print_every', type=int, default=1, help='how often to print cost')
    parser.add_argument('--optimizer', type=str, default='adagrad', choices=['adagrad', 'adam', 'sgd'])
    parser.add_argument('--expdir', type=str, default='sandbox', help='experiment directory to save files to')
    parser.add_argument('--train_frac', type=float, default=0.95, help='fraction of text file to use for training data')
    parser.add_argument('--valid_frac', type=float, default=0.05,
                        help='fraction of text file to use for validation data')
    parser.add_argument('--num_heads', type=int, default=1, help='how long input sequence to pay attention to')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--forward_only', type=bool, default=False, help="forward only = True meaning we don't train")
    # parser.add_argument('--load_model', type=str, default='',
    #                     help='load a specific epoch and run only once to inspect the model, only put directory path, will automatically load best epoch.')
    # parser.add_argument('--save_vis', type=str, default='', help='dir to save visualization, will back off to expdir')
    # parser.add_argument('--vis_style', type=str, default='real', choices=['andrej', 'real', 'histogram', 'datadump'],
    #                     help='andrej visualization, or real average activation values')
    # parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
    # parser.add_argument('--data', type=str, default='CHAR',
    #                     help='CHAR=CHAR_FILE, PTB=PTB_FILE, specify to indicate corpus')
    # parser.add_argument('--toktype', type=str, default='char', choices=['char', 'word'],
    #                     help='use word or character-level tokens')

    parser.set_defaults(batch_norm=False)

    args = parser.parse_args()
    np.random.seed(args.seed)

    args.time_steps = args.unroll

    # word_idx_map, idx_word_map = load_vocab('')
    curr = os.path.dirname(os.path.realpath(__file__))
    root = to_parent(to_parent(curr))

    loader = StoryLoader(pjoin(root, 'data/story_processed.npz'),
                         batch_size=50, src_seq_len=65,
                         tgt_seq_len=20, mode='merged')
    embedding = loader.get_w2v_embed().astype('float32')
    model = StorySeq2SeqModel(embedding, args)
