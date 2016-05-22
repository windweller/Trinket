import time
import os
import sys
import numpy as np
import theano
from theano import tensor as T
import cPickle as pickle
from word_fuel import load_data, seq_to_str
from ug_utils import _linear_params, save_model_params, load_model_params
from theano.printing import debugprint
from encdec_shared import EncoderDecoder
from opt import optimizers
from theano.printing import pydotprint
from ug_cfg import WORD_MAX_SEQ_LEN, UNK_IND, EOS_IND, WORD_SORT_K_BATCHES,\
        SRC_TRAIN_FILE, TGT_TRAIN_FILE, SRC_VOCAB_SIZE,\
        TGT_VOCAB_SIZE, SRC_VALID_FILE, TGT_VALID_FILE
from os.path import join as pjoin
from run_utils import setup_exp

# TODO
# - be able to import module separately so don't have to recompile
# - make sure calling L[i_t, :] doesn't hurt when subset_grad is true

# sanity checks to keep in mind: init, nonlinearities, optimization, transposes, regularizer parameters


def sample_output(encdec, h_ps, bdict):
    sample = ''
    s_t = np.array([EOS_IND,], dtype=np.int32)
    j = 0
    while (j == 0 or s_t != EOS_IND) and j < WORD_MAX_SEQ_LEN:
        ret = encdec.decode_step(*([s_t] + h_ps))
        p_t = ret[0]
        h_ps = ret[1:]
        h_ps = [h_p for h_p in h_ps]
        #s_t = np.argmax(np.random.multinomial(1, p_t.flatten()))
        # NOTE taking argmax instead of sampling
        s_t = np.array([np.argmax(p_t.flatten()),], dtype=np.int32)
        sample = sample + ' ' + bdict[s_t[0]]
        j = j + 1
    return sample

if __name__ == '__main__':
    import json
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_dim', type=int, default=1000, help='dimension of recurrent states and embeddings')
    parser.add_argument('--rlayers', type=int, default=1, help='number of hidden layers for RNNs')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches')
    parser.add_argument('--warm_start_lr', type=float, default=None, help='train for warm_start_iters using this lr before switching to larger lr')
    parser.add_argument('--warm_start_iters', type=int, default=100, help='see warm_start_lr')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--lr_decay_after', type=int, default=3, help='epoch after which to decay')
    #parser.add_argument('--max_norm_elemwise', type=float, default=0.1, help='element-wise norm at which to clip gradients')
    parser.add_argument('--max_norm', type=float, default=5.0, help='norm at which to rescale gradients')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout (fraction of units randomly dropped on non-recurrent connections)')
    parser.add_argument('--src_vocab_size', type=int, default=SRC_VOCAB_SIZE, help='source vocabulary size')
    parser.add_argument('--tgt_vocab_size', type=int, default=TGT_VOCAB_SIZE, help='source vocabulary size')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train')
    parser.add_argument('--print_every', type=int, default=1, help='how often to print cost')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=optimizers)
    parser.add_argument('--reverse', action='store_true', help='reverse source input sentence')
    parser.add_argument('--expdir', type=str, default='sandbox', help='experiment directory to save files to')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume starting at this epoch (expdir must have saved models, and this will also load previous opts.json for configuration)')
    args = parser.parse_args()
    args.max_seq_len = WORD_MAX_SEQ_LEN; args.sort_k_batches = WORD_SORT_K_BATCHES

    logger, opts = setup_exp(args)

    logger.info('loading data...')
    train_stream, src_dict, tgt_dict = load_data(SRC_TRAIN_FILE, TGT_TRAIN_FILE, args.batch_size, WORD_SORT_K_BATCHES, training=True)
    logger.info('done loading data')
    src_bdict = {v: k for k, v in src_dict.iteritems()}
    tgt_bdict = {v: k for k, v in tgt_dict.iteritems()}
    valid_stream, _, _ = load_data(SRC_VALID_FILE, TGT_VALID_FILE, args.batch_size, WORD_SORT_K_BATCHES, training=False)

    if args.resume_epoch:
        assert (args.resume_epoch > 0)
        start_epoch = args.resume_epoch
        logger.info('loading model...')
        encdec = EncoderDecoder(args)
        load_model_params(encdec, pjoin(args.expdir, 'model_epoch%d.h5' % (args.resume_epoch - 1)))
        logger.info('done loading model')
    else:
        start_epoch = 0
        encdec = EncoderDecoder(args)
        logger.info('# params: %d' % encdec.nparams)

    # print graph for debugging (encdec.train should be optimized,
    # print encdec.cost for unoptimized)
    #pydotprint(encdec.train, outfile=pjoin(args.expdir, 'model.png'), var_with_name_simple=True)

    lr = args.lr
    for epoch in xrange(start_epoch, args.epochs):
        if epoch >= args.lr_decay_after:
            lr = lr * args.lr_decay
            logger.info('decaying learning rate to: %f' % lr)
        exp_length = None
        exp_cost = None

        it = 0
        for ss, sm, ts, tm in train_stream.get_epoch_iterator():
            if epoch == 0 and args.warm_start_lr:
                lr = args.warm_start_lr if it < args.warm_start_iters else args.lr
            try:
                tic = time.time()
                it = it + 1
                ss, ts = ss.astype(np.int32), ts.astype(np.int32)
                sm, tm = sm.astype(np.bool), tm.astype(np.bool)
                rss = reverse_sent(ss, sm)
                cost, grad_norm, param_norm = encdec.train(ss, sm, rss, ts, tm, args.dropout, lr, args.max_norm)
                norm_ratio = grad_norm / param_norm
                # normalize by average length when printing
                # note exclude the first token for computing cost
                lengths = np.sum(tm[:, 1:], axis=1)
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)
                if not exp_cost:
                    exp_cost = cost
                    exp_length = mean_length
                else:
                    exp_cost = 0.99*exp_cost + 0.01*cost
                    exp_length = 0.99*exp_length + 0.01*mean_length
                cost = cost / mean_length
                toc = time.time()
                if (it + 1) % args.print_every == 0:
                    logger.info('epoch %d, iter %d, cost %f, expcost %f, grad/param norm %f, batch time %f, length mean/stdev %f/%f' %\
                            (epoch + 1, it, cost, exp_cost/exp_length, norm_ratio, toc - tic, mean_length, std_length))
            except KeyboardInterrupt:
                confirm = raw_input('Are you sure you want to quit in middle of training? (Enter "yes" to quit)')
                if confirm == 'yes':
                    sys.exit()
                else:
                    continue

        logger.info('saving model')
        save_model_params(encdec, pjoin(args.expdir, 'model_epoch%d.h5' % epoch))

        # run on validation
        valid_costs = []
        valid_lengths = []
        for ss, sm, ts, tm in valid_stream.get_epoch_iterator():
            ss, ts = ss.astype(np.int32), ts.astype(np.int32)
            sm, tm = sm.astype(np.bool), tm.astype(np.bool)
            rss = reverse_sent(ss, sm)
            # NOTE batches sometimes different sizes
            # NOTE set dropout rate to 0
            cost = encdec.test(ss, sm, rss, ts, tm)
            length = np.sum(tm[:, 1:])
            valid_costs.append(cost * tm.shape[0])
            valid_lengths.append(length)

            # sample / generate

            if len(valid_costs) == 1:  # NOTE only generate from first batch
                for k in xrange(ss.shape[0]):
                    # extract single example
                    ssk, smk, tsk, tmk = ss[k:k+1], sm[k:k+1], ts[k:k+1], tm[k:k+1]
                    rssk = reverse_sent(ssk, smk)
                    sslen = np.sum(smk)
                    h_t = encdec.encode(ssk, rssk, smk)
                    #h_t = [x[0] for x in h_t]
                    sample = sample_output(encdec, h_t, tgt_bdict)
                    logger.info(seq_to_str(ssk[0, 0:sslen].tolist(), src_bdict) + ' | ' + sample)

        logger.info('validation cost: %f' %\
                (sum(valid_costs) / float(sum(valid_lengths))))
