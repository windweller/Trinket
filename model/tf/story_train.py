import tensorflow as tf
from util import load_vocab
import json
import argparse
import logging
import numpy as np
import story_model
from trident_cfg import STORY_DATA_PATH
import sys

flags = tf.flags

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("src_steps", 65, "Time steps.")
tf.app.flags.DEFINE_integer("tgt_steps", 20, "Time steps.")
tf.app.flags.DEFINE_integer("label_size", 2, "Size of the label.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
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


def create_model(session, vocab_size, embedding, forward_only):
    model = story_model.StoryModel(vocab_size, FLAGS.label_size, FLAGS.size, FLAGS.num_layers,
                                   FLAGS.batch_size, FLAGS.learning_rate,
                                   FLAGS.learning_rate_decay_factor,
                                   FLAGS.dropout, embedding, FLAGS.src_steps, FLAGS.tgt_steps,
                                   FLAGS.mode, FLAGS.max_gradient_norm, forward_only=forward_only)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())

    return model

def train():
  """Train a translation model using NLC data."""
  # Prepare NLC data.
  print("Preparing NLC data in %s" % FLAGS.data_dir)

  x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
    FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size,
    tokenizer=get_tokenizer(FLAGS))
  vocab, _ = nlc_data.initialize_vocabulary(vocab_path)
  vocab_size = len(vocab)
  print("Vocabulary size: %d" % vocab_size)

  with tf.Session() as sess:
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, vocab_size, False)

    if False:
      tic = time.time()
      params = tf.trainable_variables()
      num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
      toc = time.time()
      print ("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    epoch = 0
    previous_losses = []
    while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
      epoch += 1
      current_step = 0
      exp_cost = None
      exp_length = None
      exp_norm = None

      ## Train
      for source_tokens, source_mask, target_tokens, target_mask in PairIter(x_train, y_train, FLAGS.batch_size, FLAGS.num_layers):
        # Get a batch and make a step.
        tic = time.time()

        grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)

        toc = time.time()
        iter_time = toc - tic
        current_step += 1

        lengths = np.sum(target_mask, axis=0)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        if not exp_cost:
          exp_cost = cost
          exp_length = mean_length
          exp_norm = grad_norm
        else:
          exp_cost = 0.99*exp_cost + 0.01*cost
          exp_length = 0.99*exp_length + 0.01*mean_length
          exp_norm = 0.99*exp_norm + 0.01*grad_norm

        cost = cost / mean_length

        if current_step % FLAGS.print_every == 0:
          print('epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, batch time %f, length mean/std %f/%f' %
                (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, iter_time, mean_length, std_length))

      ## Checkpoint
      checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)

      valid_costs, valid_lengths = [], []
      for source_tokens, source_mask, target_tokens, target_mask in PairIter(x_dev, y_dev, FLAGS.batch_size, FLAGS.num_layers):
        cost, _ = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
      valid_cost = sum(valid_costs) / float(sum(valid_lengths))

      print("Epoch %d Validation cost: %f" % (epoch, valid_cost))

      previous_losses.append(valid_cost)
      if len(previous_losses) > 2 and valid_cost > max(previous_losses[-3:]):
        sess.run(model.learning_rate_decay_op)
      sys.stdout.flush()


if __name__ == '__main__':
    np.random.seed(FLAGS.seed)
    word_idx_map, idx_word_map = load_vocab('')
