import random

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = tf.reshape(self.hs, [-1, num_units])
                phi_hs2d = tanh(rnn_cell.linear(hs2d, num_units, True, 1.0))
                self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell.linear(gru_out, self._num_units, True, 1.0))
            weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
            weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
            weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
            context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell.linear([context, gru_out], self._num_units, True, 1.0))
            self.attn_map = tf.squeeze(tf.slice(weights, [0, 0, 0], [-1, -1, 1]))
            return (out, out)


class StoryModel(object):
    def __init__(self, vocab_size, label_size, size, num_layers, batch_size, learning_rate,
                 learning_rate_decay_factor, dropout, embedding, src_steps, tgt_steps,
                 mode='sq2sq',
                 max_gradient_norm=5.0, forward_only=False):

        self.size = size
        self.mode = mode
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embedding = embedding
        self.src_steps = src_steps
        self.tgt_steps = tgt_steps
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.keep_prob = 1.0 - dropout
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.source_tokens = tf.placeholder(tf.int32, shape=[None, self.src_steps], name='srcInput')
        self.target_tokens = tf.placeholder(tf.int32, shape=[None, self.tgt_steps], name='targetInput')
        self.label_placeholder = tf.placeholder(tf.float32, shape=[None, self.label_size])

        self.decoder_state_input, self.decoder_state_output = [], []
        self.tgt_encoder_state_input, self.tgt_encoder_state_output = [], []

        for i in xrange(num_layers):
            self.decoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))
            self.tgt_encoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))

        self.setup_embeddings()
        self.setup_encoder()
        self.setup_decoder()
        if mode == 'sq2sq':
            self.setup_label_loss()
        else:
            raise NotImplementedError

        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.AdamOptimizer(self.learning_rate)

            gradients = tf.gradients(self.losses, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.gradient_norm = tf.global_norm(clipped_gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            self.encoder_inputs = embedding_ops.embedding_lookup(self.embedding, self.source_tokens)
            self.decoder_inputs = embedding_ops.embedding_lookup(self.embedding, self.target_tokens)

    def setup_encoder(self):
        self.encoder_cell = rnn_cell.GRUCell(self.size, input_size=self.embedding.shape[1])
        # we took out the mask, so this no longer is the pyramid encoder
        with vs.variable_scope("Encoder"):
            inp = self.encoder_inputs
            out = None
            for i in xrange(self.num_layers):
                with vs.variable_scope("EncoderCell%d" % i) as scope:
                    # out, _ = self.bidirectional_rnn(self.encoder_cell, self.dropout(inp),
                    #                                 self.src_steps, scope=scope)
                    out, _ = rnn.dynamic_rnn(self.encoder_cell, self.dropout(inp), time_major=False,
                                             dtype=dtypes.float32,
                                             sequence_length=self.src_steps, scope=scope)
            self.encoder_output = out

    def bidirectional_rnn(self, cell, inputs, lengths, scope=None):
        name = scope.name or "BiRNN"
        # Forward direction
        with vs.variable_scope(name + "_FW") as fw_scope:
            output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=False, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=fw_scope)
        # Backward direction
        with vs.variable_scope(name + "_BW") as bw_scope:
            output_bw, output_state_bw = rnn.dynamic_rnn(cell, inputs, time_major=False, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=bw_scope)

        output_bw = tf.reverse_sequence(output_bw, tf.to_int64(lengths), seq_dim=0, batch_dim=1)

        outputs = output_fw + output_bw
        output_state = output_state_fw + output_state_bw

        return (outputs, output_state)

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def setup_decoder(self):
        """
        This sets up a decoder

        but we may need a double-encoder

        Returns
        -------
        """
        if self.num_layers > 1:
            self.decoder_cell = rnn_cell.GRUCell(self.size, input_size=self.embedding[1])
        self.attn_cell = GRUCellAttn(self.size, self.encoder_output, scope="DecoderAttnCell")

        out = self.decoder_inputs

        with vs.variable_scope("Decoder"):
            inp = self.decoder_inputs
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    out, state_output = rnn.dynamic_rnn(self.decoder_cell, self.dropout(inp), time_major=False,
                                                        dtype=dtypes.float32, sequence_length=self.tgt_steps,
                                                        scope=scope, initial_state=self.decoder_state_input[i])
                    inp = out
                    self.decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                out, state_output = rnn.dynamic_rnn(self.attn_cell, self.dropout(inp), time_major=False,
                                                    dtype=dtypes.float32, sequence_length=self.tgt_steps,
                                                    scope=scope, initial_state=self.decoder_state_input[i + 1])
                self.decoder_output = out
                self.decoder_state_output.append(state_output)

    def setup_target_encoder(self):
        """
        This sets up an encoder that works on
        target sentence and produce a single label in the end
        encoder has attentions

        Returns
        -------
        """
        if self.num_layers > 1:
            self.tgt_encoder_cell = rnn_cell.GRUCell(self.size, input_size=self.embedding[1])
        self.attn_cell = GRUCellAttn(self.size, self.encoder_output, scope="EncoderAttnCell")

        out = self.decoder_inputs

        with vs.variable_scope("TgtEncoder"):
            inp = self.decoder_inputs
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("TgtEncoderCell%d" % i) as scope:
                    out, state_output = rnn.dynamic_rnn(self.tgt_encoder_cell, self.dropout(inp), time_major=False,
                                                        dtype=dtypes.float32, sequence_length=self.tgt_steps,
                                                        scope=scope, initial_state=self.tgt_encoder_state_output[i])
                    inp = out
                    self.tgt_encoder_state_output.append(state_output)

            with vs.variable_scope("TgtEncoderAttnCell") as scope:
                out, state_output = rnn.dynamic_rnn(self.attn_cell, self.dropout(inp), time_major=False,
                                                    dtype=dtypes.float32, sequence_length=self.tgt_steps,
                                                    scope=scope, initial_state=self.tgt_encoder_state_output[i + 1])
                self.tgt_encoder_output = out
                self.tgt_encoder_state_output.append(state_output)

    def setup_label_loss(self):
        with vs.variable_scope("LabelLogistic"):
            doshape = tf.shape(self.decoder_output)
            T, batch_size = doshape[0], doshape[1]

            # [batch_size, cell.state_size]
            # decoder_output: [batch_size, time_step, cell.state_size]
            last_state = self.decoder_output[:, -1, :]

            # projecting to label space
            # [batch_size, label_size]
            logits = rnn_cell.linear(last_state, self.label_size, True, 1.0)
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits, self.label_placeholder)
            self.predictions = logits

    def setup_generation_loss(self):
        with vs.variable_scope("Logistic"):
            doshape = tf.shape(self.decoder_output)
            T, batch_size = doshape[0], doshape[1]
            do2d = tf.reshape(self.decoder_output, [-1, self.size])
            logits2d = rnn_cell.linear(do2d, self.vocab_size, True, 1.0)
            outputs2d = tf.nn.log_softmax(logits2d)
            self.outputs = tf.reshape(outputs2d, tf.pack([T, batch_size, self.vocab_size]))

            targets_no_GO = tf.slice(self.target_tokens, [1, 0], [-1, -1])
            # easier to pad target/mask than to split decoder input since tensorflow does not support negative indexing
            labels1d = tf.reshape(tf.pad(targets_no_GO, [[0, 1], [0, 0]]), [-1])
            losses1d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2d, labels1d)
            losses2d = tf.reshape(losses1d, tf.pack([T, batch_size]))
            self.losses = tf.reduce_sum(losses2d) / tf.to_float(batch_size)

    def set_default_decoder_state_input(self, input_feed, batch_size):
        default_value = np.zeros([batch_size, self.size])
        for i in xrange(self.num_layers):
            input_feed[self.decoder_state_input[i]] = default_value

    def train(self, session, source_tokens, target_tokens, labels=None, mode='sq2sq'):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.target_tokens] = target_tokens
        if mode == 'sq2sq':
            input_feed[self.label_placeholder] = labels
        self.set_default_decoder_state_input(input_feed, target_tokens.shape[1])

        output_feed = [self.updates, self.gradient_norm, self.losses, self.param_norm, self.predictions]

        outputs = session.run(output_feed, input_feed)

        # last one is predictions
        return outputs[1], outputs[2], outputs[3], outputs[4]

    def test(self, session, source_tokens, target_tokens, labels=None, mode='sq2sq'):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.target_tokens] = target_tokens
        if mode == 'sq2sq':
            input_feed[self.label_placeholder] = labels
        self.set_default_decoder_state_input(input_feed, target_tokens.shape[1])

        output_feed = [self.losses, self.outputs]

        outputs = session.run(output_feed, input_feed)

        return outputs[0], outputs[1]

if __name__ == '__main__':
    from trident_cfg import STORY_DATA_PATH, VOCAB_PATH, EMBED_PATH
    from util import load_vocab
    from os.path import join as pjoin
    from data.story_loader import StoryLoader

    flags = tf.flags

    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_integer("src_steps", 65, "Time steps.")
    tf.app.flags.DEFINE_integer("tgt_steps", 20, "Time steps.")
    tf.app.flags.DEFINE_integer("label_size", 2, "Size of the label.")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
    tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
    tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
    tf.app.flags.DEFINE_float("dropout", 0.0, "Fraction of units randomly dropped on non-recurrent connections.")
    tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use during training.")
    tf.app.flags.DEFINE_integer("epochs", 1, "Number of epochs to train.")
    tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
    tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
    tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training weights are saved in it.")
    tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
    tf.app.flags.DEFINE_integer('seed', 1234, 'random seed')
    tf.app.flags.DEFINE_string('mode', 'sq2sq', 'Choose [sq2sq, att, gen]')
    tf.app.flags.DEFINE_string('gpu', '3', 'Choose GPU ID: [0,1,2,3]')
    tf.app.flags.DEFINE_string('embed', 'w2v', 'Choose embedding: [w2v, glove50, glove100, glove200, glove300]')

    word_idx_map, idx_word_map = load_vocab(VOCAB_PATH)
    vocab_size = len(idx_word_map)

    loader = StoryLoader(STORY_DATA_PATH,
                        batch_size=50, src_seq_len=65,
                        tgt_seq_len=20, mode='merged')

    if FLAGS.embed == 'w2v':
        embedding = loader.get_w2v_embed().astype('float32')

    model = StoryModel(vocab_size, FLAGS.label_size, FLAGS.size, FLAGS.num_layers,
                                   FLAGS.batch_size, FLAGS.learning_rate,
                                   FLAGS.learning_rate_decay_factor,
                                   FLAGS.dropout, embedding, FLAGS.src_steps, FLAGS.tgt_steps,
                                   FLAGS.mode, FLAGS.max_gradient_norm, forward_only=False)