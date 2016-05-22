import numpy as np
import theano
from theano import tensor as T
from ug_utils import floatX, Dropout
from rnn import (RNN, SequenceLogisticRegression, GRULayer, GRULayerAttention, LSTMLayer,
                 LayerWrapper, seq_cat_crossent, Downscale)
from opt import get_opt_fn
from ug_utils import (glorot_init, norm_init, uniform_init,
                      get_sequence_dropout_mask, _linear_params)
from opt import optimizers
from run_utils import setup_exp

Unit = {'gru': GRULayer, 'lstm': LSTMLayer}
Bidir = False



class StoryModelSeq2Seq(object):
    def __init__(self, args):
        self.args = args
        x = T.imatrix('x')  # (time_step, batch_size) We do x.T before put it in
        y = T.imatrix('y')  # (label_size, )

class UnitInit(object):

    def __init__(self, x, mask, seqmask, x_dim, outputs_info, args, suffix='', backwards=False):
        if not Bidir:
            l = Unit[args.unit](x, mask, seqmask, x_dim, outputs_info, args, 'f'+suffix)
            self.out = l.out
            self.params = l.params
            if args.unit == 'lstm':
                self.cell = l.cell
            self._step = l._step
            if args.load_model != '':
                self.activations = l.activations
        else:
            self.U = _linear_params(args.rnn_dim * 2, args.rnn_dim, 'u%s' % suffix, act=T.nnet.relu, bias=False)
            # we are reusing same seqmask for forward/backward pass
            flayer = Unit[args.unit](x, mask, seqmask, x_dim, outputs_info, args, 'f'+suffix, backwards=False)
            blayer = Unit[args.unit](x, mask, seqmask, x_dim, outputs_info, args, 'b'+suffix, backwards=True)

            # (T, N=batch_size, 2*D=rnn_dim)
            fb_out = T.concatenate((flayer.out, blayer.out), axis=2)

            self._step = flayer._step  # this is for sampling
            # we only sample forward pass

            self.out = T.dot(fb_out, self.U)
            self.params = flayer.params + blayer.params

            self.params.append(self.U)


if __name__ == '__main__':
    import json
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_dim', type=int, default=1000, help='dimension of recurrent states and embeddings')
    parser.add_argument('--rlayers', type=int, default=1, help='number of hidden layers for RNNs')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batches')
    parser.add_argument('--warm_start_lr', type=float, default=None,
                        help='train for warm_start_iters using this lr before switching to larger lr')
    parser.add_argument('--warm_start_iters', type=int, default=100, help='see warm_start_lr')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--lr_decay_after', type=int, default=3, help='epoch after which to decay')
    parser.add_argument('--max_norm', type=float, default=5.0, help='norm at which to rescale gradients')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout (fraction of units randomly dropped on non-recurrent connections)')
    parser.add_argument('--src_steps', type=int, default=65, help='source sequence length')
    parser.add_argument('--tgt_steps', type=int, default=20, help='target sequence length')
    parser.add_argument('--label_size', type=int, default=2, help='number of labels that we need to predict')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train')
    parser.add_argument('--print_every', type=int, default=1, help='how often to print cost')
    parser.add_argument('--optimizer', type=str, default='adam', choices=optimizers)
    parser.add_argument('--reverse', action='store_true', help='reverse source input sentence')
    parser.add_argument('--expdir', type=str, default='sandbox', help='experiment directory to save files to')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='resume starting at this epoch (expdir must have saved models, and this will also load previous opts.json for configuration)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--embed', type=str, default='w2v', choices=['w2v', 'glove50', 'glove100',
                                                                     'glove200', 'glove300'], help='which embedding to load in')
    parser.add_argument('--mode', type=str, default='sq2sq', choices=['sq2sq', 'att', 'gen'], help='which embedding to load in')

    args = parser.parse_args()

    logger, opts = setup_exp(args)
    logger.info(args)

