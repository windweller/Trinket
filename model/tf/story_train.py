import tensorflow as tf
from util import load_vocab
import json
import argparse
import logging

flags = tf.flags

# flags.DEFINE_string()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_dim', type=int, default=256, help='dimension of recurrent states')
    parser.add_argument('--rlayers', type=int, default=2, help='number of hidden layers for RNNs')
    parser.add_argument('--unroll', type=int, default=50, help='number of time steps to unroll for')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batches')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    # parser.add_argument('--lr_decay_after', type=int, default=10, help='epoch after which to decay')
    parser.add_argument('--lr_decay_threshold', type=float, default=0.01,
                        help='begin decaying learning rate if diff of prev 2 validation costs less than thresohld')
    parser.add_argument('--max_lr_decays', type=int, default=8, help='maximum number of times to decay learning rate')
    # parser.add_argument('--max_norm_elemwise', type=float, default=0.1, help='norm at which to clip gradients elementwise')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout (fraction of units randomly dropped on non-recurrent connections)')
    parser.add_argument('--recdrop', action='store_true',
                        help='use dropout on recurrent updates if True, use stocdrop if False')
    parser.add_argument('--stocdrop', type=float, default=0.0, help='use to set droprate for stocdrop or recdrop')
    parser.add_argument('--worddrop', type=float, default=0.0, help='use to set word droprate ')
    parser.add_argument('--dropstrat', choices=['regular', 'frequency'], default='regular')
    parser.add_argument('--freqscale', type=float, default=0.1,
                        help='value to scale frequencies by for frequency dropout mask')
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
    parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
    parser.add_argument('--ortho', dest='ortho', action='store_true', help='Orthogonal Initialization')
    parser.add_argument('--data', type=str, default='CHAR',
                        help='CHAR=CHAR_FILE, PTB=PTB_FILE, specify to indicate corpus')
    parser.add_argument('--toktype', type=str, default='char', choices=['char', 'word'],
                        help='use word or character-level tokens')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--load_model', type=str, default='',
                        help='load a specific epoch and run only once to inspect the model, only put directory path, will automatically load best epoch.')
    parser.add_argument('--save_vis', type=str, default='', help='dir to save visualization, will back off to expdir')
    parser.add_argument('--vis_style', type=str, default='real', choices=['andrej', 'real', 'histogram', 'datadump'],
                        help='andrej visualization, or real average activation values')

    parser.set_defaults(batch_norm=False)

    args = parser.parse_args()
    np.random.seed(args.seed)

    word_idx_map, idx_word_map = load_vocab('')
