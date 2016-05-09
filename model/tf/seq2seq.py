"""
A seq2seq model for
story training
"""

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
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops

import numpy as np

Unit = {'gru': tf.nn.rnn_cell.GRUCell,
        'lstm': tf.nn.rnn_cell.BasicLSTMCell}


def attention_decoder_with_embedding(decoder_inputs, initial_state, attention_states,
                                     cell, num_symbols, num_heads=1,
                                     output_size=None, output_projection=None,
                                     feed_previous=False,
                                     update_embedding_for_previous=True,
                                     dtype=dtypes.float32, scope=None,
                                     initial_state_attention=False):
    pass


def attention_seq2seq_with_embedding(encoder_inputs, decoder_inputs, num_decoder_symbols,
                                     cell, num_heads=1, output_projection=None,
                                     feed_previous=False, dtype=dtypes.float32,
                                     scope=None, initial_state_attention=False):
    """
    Our twist on "embedding_attention_seq2seq"

    but with pre-initialized embeddings, the
    rest are the same.

    Returns
    -------
    """
    with vs.variable_scope(scope or "embedding_attention_seq2seq"):
        # Encoder.
        encoder_cell = rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.rnn(
            encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        # We don't consider feed_previous being a Tensor,
        # check original implementation for feed_previous being tensor case
        if isinstance(feed_previous, bool):
            return attention_decoder_with_embedding(
                decoder_inputs, encoder_state, attention_states, cell,
                num_decoder_symbols, num_heads=num_heads,
                output_size=output_size, output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)
        else:
            raise NotImplementedError


class StorySeq2SeqModel(object):
    """
    An adapatation of seq2seq model of tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py
    Diff:

    We implement bi-rnn
    We also don't do buckets (since it's hard to compare)
    """

    def __init__(self, args):
        """

        Parameters
        ----------
        args: object (could be FLAGS)

            hidden_dim: int
                number of units in each layer of the model. (same as 'size' in seq2seq)
            vocab_size: int
                number of vocab in the corpus (maybe split it to target_vocab_size
                and src_vocab_size is better?)
            num_layers: int
                number of layers in the model.
            max_gradient_norm: int
                gradients will be clipped to maximally this norm. (set to 5.0 in translate.py)
            learning_rate: float
            earning_rate_decay_factor: float
                decay learning rate by this much when needed.
            unit: {'lstm', 'gru'}
                We can consider adding bi-lstm/bi-gru (bi-rnn) in
            num_samples: int
                number of samples for sampled softmax (default: 512)
            forward_only: boolean
                if set, we do not construct the backward pass in the model. (no learning)
        """
        self.args = args
        self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * args.earning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        if args.num_samples != 0:
            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [args.size, args.vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [args.vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                with tf.device("/cpu:0"):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, args.num_samples,
                                                      args.vocab_size)

            softmax_loss_function = sampled_loss

        # multilayer
        cell = Unit[args.unit](args.hidden_dim)
        if args.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
