import numpy as np
import theano
import time
from theano import tensor as T
from ug_utils import floatX, Dropout, save_model_params, load_model_params
from rnn import (RNN, SequenceLogisticRegression, LogisticRegression, GRULayer, GRULayerAttention,
                 LayerWrapper, seq_cat_crossent, Downscale, cross_entropy)
from encdec_shared import BiRNNEncoder, reverse_sent, RNNEncoder, RNNDecoderAttention
from opt import get_opt_fn
from ug_utils import (glorot_init, norm_init, uniform_init,
                      get_sequence_dropout_mask, _linear_params, get_parent)
from opt import optimizers
from run_utils import setup_exp, get_logger
from trident_cfg import STORY_DATA_PATH, VOCAB_PATH, EMBED_PATH
from util import load_vocab, decode_sentences
from os.path import join as pjoin
from data.story_loader import StoryLoader
import cPickle as pickle


def prep_tgt_sens(tgt1, tgt2, args):
    # tgt1: (time_step, N, embed_size)
    # return vector, and input_size (for first-layer in RNN Layers)

    if args.vector_mode == 'sub':
        return tgt1 - tgt2, args.input_size
    elif args.vector_mode == 'concat':
        concatenated = T.concatenate([tgt1, tgt2], axis=2)
        return concatenated, args.input_size * 2
    elif args.vector_mode == 'tgt1':
        return tgt1, args.input_size
    else:
        raise NotImplementedError


class RNNEncoderAttention(RNN):
    # used for target encoding
    def __init__(self, encoder, target_sqn, target2_sqn, label, mask, L_dec, pdrop, args):
        # target_sqn: (time_step, N)
        self.hs = encoder.hs

        # NOTE just use this so only last layer uses attention
        def layer_init(attention):
            if not attention:
                return GRULayer
            else:
                return lambda *largs, **kwargs: GRULayerAttention(self.hs, *largs, **kwargs)

        # initial states
        outputs_info = encoder.out
        rlayers = list()

        target_inp1 = L_dec[target_sqn]
        target_inp2 = L_dec[target2_sqn]

        inp, self.input_size = prep_tgt_sens(target_inp1, target_inp2, args)
        # inp = L_dec[target_sqn]

        attention = args.rlayers == 1
        # exclude last prediction
        seqmask = get_sequence_dropout_mask((target_sqn.shape[0], target_sqn.shape[1], L_dec.shape[1]), pdrop)
        inplayer = layer_init(attention)(inp.astype(floatX), mask, seqmask, self.input_size,
                                         outputs_info[0], args, suffix='tgtenc0')
        rlayers.append(inplayer)
        for k in xrange(1, args.rlayers):
            attention = (args.rlayers == k + 1)
            seqmask = get_sequence_dropout_mask((target_sqn.shape[0], target_sqn.shape[1], args.rnn_dim), pdrop)
            rlayer = layer_init(attention)(Dropout(rlayers[-1].out, pdrop).out, mask,
                                           seqmask, args.rnn_dim, outputs_info[k], args, suffix='dec%d' % k)
            rlayers.append(rlayer)
        # we only classify the final state
        olayer = LogisticRegression(Dropout(rlayers[-1].out, pdrop).out[-1, :, :], args.rnn_dim,
                                    args.label_size)
        cost = cross_entropy(olayer.out, label, normalize=False)
        super(RNNEncoderAttention, self).__init__(rlayers, olayer, cost)


class RNNTargetEncoder(RNN):
    def __init__(self, init_states, target_sqn, target2_sqn, label, mask, L_dec, pdrop, args, suffix_prefix='tgtEnc'):
        # exactly like above but without attention
        # target_sqn: (time_step, N)

        # initial states
        # [(batch_size, rnn_dim)]
        # outputs_info = [T.zeros((target_sqn.shape[1], args.rnn_dim)).astype(floatX)]
        outputs_info = init_states  # should be a list of things...(check RNNDecoder)
        rlayers = list()

        target_inp1 = L_dec[target_sqn]
        target_inp2 = L_dec[target2_sqn]

        inp, self.input_size = prep_tgt_sens(target_inp1, target_inp2, args)

        # exclude last prediction
        seqmask = get_sequence_dropout_mask((target_sqn.shape[0], target_sqn.shape[1], self.input_size), pdrop)
        inplayer = GRULayer(inp.astype(floatX), mask, seqmask, self.input_size, outputs_info[0],
                            args, suffix='%s0' % suffix_prefix, backwards=False)

        rlayers.append(inplayer)
        for k in xrange(1, args.rlayers):
            inp = rlayers[-1].out
            seqmask = get_sequence_dropout_mask((target_sqn.shape[0], target_sqn.shape[1], args.rnn_dim), pdrop)
            rlayer = GRULayer(Dropout(inp, pdrop).out, mask, seqmask, args.rnn_dim,
                              outputs_info[k], args, suffix='%s%d' % (suffix_prefix, k), backwards=False)
            rlayers.append(rlayer)

        # we only classify the final state
        olayer = LogisticRegression(Dropout(rlayers[-1].out, pdrop).out[-1, :, :], args.rnn_dim,
                                    args.label_size)
        cost = cross_entropy(olayer.out, label, normalize=False)
        super(RNNTargetEncoder, self).__init__(rlayers, olayer, cost)


class StoryModelSeq2Seq(object):
    def __init__(self, args, embedding, attention=False):
        self.args = args

        self.embedding = theano.shared(embedding, 'embedding', borrow=True)

        labels = T.ivector('labels')  # (label_size, )

        self.rnn_dim = args.rnn_dim
        self.rlayers = args.rlayers
        self.attention = attention

        lr = T.scalar(dtype=floatX)
        pdrop = T.scalar(dtype=floatX)
        max_norm = T.scalar(dtype=floatX)
        src_sent = T.imatrix('src_sent')  # (batch_size, time_step) We do x.T before put it in
        rev_src_sent = T.imatrix('rev_src_sent')  # for reverse, use reverse_sent()
        tgt_sent = T.imatrix('tgt_sent')
        space_mask = T.bmatrix('space_mask')

        src_mask = T.ones_like(src_sent).astype(floatX)  # this is used to drop words? Now we don't
        tgt_mask = T.ones_like(tgt_sent).astype(floatX)  # this is used to drop words? Now we don't

        # we are taking a 2nd target sentence for
        # target encoder
        # we merge this with tgt_sent inside Encoder
        tgt2_sent = T.imatrix('tgt2_sent')  # second target sentence (N, T)

        if args.bidir:
            print('Using bidirectional GRU encoder')
            # x, xr, mask, space_mask, L_enc, pdrop, args
            self.encoder = BiRNNEncoder(src_sent.T, rev_src_sent.T, src_mask.T, space_mask.T, self.embedding, pdrop,
                                        args)
        else:
            print 'Using unidirectional GRU encoder'
            self.encoder = RNNEncoder(src_sent.T, src_mask.T, space_mask.T, self.embedding, pdrop, args)

        if attention:
            # encoder, target_sqn, target2_sqn, label, mask, L_dec, pdrop, args
            self.tgt_encoder = RNNEncoderAttention(self.encoder, tgt_sent.T, tgt2_sent.T, labels,
                                                   tgt_mask.T, self.embedding, pdrop,
                                                   args)
            hs = self.tgt_encoder.hs
        else:
            # init_states, target_sqn, target2_sqn, label, mask, L_dec, pdrop
            self.tgt_encoder = RNNTargetEncoder(self.encoder.out, tgt_sent.T, tgt2_sent.T, labels,
                                                tgt_mask.T, self.embedding, pdrop,
                                                args)

        if args.pretrain:
            # encoder, y, mask, L_dec, pdrop, args
            # tgt_sent is the real last sentence
            self.tgt_decoder = RNNDecoderAttention(self.encoder, tgt_sent.T, tgt_mask.T, self.embedding,
                                                   pdrop, args)
            self.pretrain_cost = self.tgt_decoder.cost
            self.pretrain_params = self.encoder.params + self.tgt_decoder.params

            # notice we are just skipping grad_params for convenience
            self.pretrain_updates, self.pretrain_grad_norm, self.pretrain_param_norm = get_opt_fn(args.optimizer)(
                self.pretrain_cost,
                self.pretrain_params,
                lr, max_norm=max_norm)
            self.pretrain_nparams = np.sum([np.prod(p.shape.eval()) for p in self.pretrain_params])

            self.pretrain = theano.function(
                inputs=[src_sent, tgt_sent, pdrop, lr, max_norm],
                outputs=[self.pretrain_cost, self.pretrain_grad_norm, self.pretrain_param_norm],
                updates=self.pretrain_updates,
                on_unused_input='warn',
                allow_input_downcast=True
            )

        # cost, parameters, grads, updates

        self.cost = self.tgt_encoder.cost
        self.params = self.encoder.params + self.tgt_encoder.params  # + self.embedding (not trainable)

        self.grad_params = self.params
        self.updates, self.grad_norm, self.param_norm = get_opt_fn(args.optimizer)(self.cost, self.grad_params,
                                                                                   lr, max_norm=max_norm)

        self.nparams = np.sum([np.prod(p.shape.eval()) for p in self.params])

        # functions
        if args.bidir:
            self.train = theano.function(
                inputs=[src_sent, rev_src_sent, tgt_sent, tgt2_sent, labels,
                        pdrop, lr, max_norm],
                outputs=[self.cost, self.grad_norm, self.param_norm, self.tgt_encoder.olayer.y_pred],
                updates=self.updates,
                on_unused_input='warn',
                allow_input_downcast=True
            )
            self.test = theano.function(
                inputs=[src_sent, rev_src_sent, tgt_sent, tgt2_sent, labels, theano.In(pdrop, value=0.0)],
                outputs=[self.cost, self.tgt_encoder.olayer.y_pred],
                updates=None,
                on_unused_input='warn'
            )
        else:
            self.train = theano.function(
                inputs=[src_sent, tgt_sent, tgt2_sent, labels, pdrop, lr, max_norm],
                outputs=[self.cost, self.grad_norm, self.param_norm, self.tgt_encoder.olayer.y_pred],
                updates=self.updates,
                on_unused_input='warn',
                allow_input_downcast=True
            )
            self.test = theano.function(
                inputs=[src_sent, tgt_sent, tgt2_sent, labels, theano.In(pdrop, value=0.0)],
                outputs=[self.cost, self.tgt_encoder.olayer.y_pred],
                updates=None,
                on_unused_input='warn'
            )

        if args.pretrain:
            # sample ending:
            # 1. encode a story (in the validation set)
            # 2. sample with a decoder
            pass


def decode(t, i, idx_word_map):
    l = []
    for e in t:
        l.append(decode_sentences(e[i], idx_word_map))
    return l


def save_result(loc, x, y_1, y_2, real_label, pred_label, idx_word_map):
    import csv
    with open(loc, 'a') as f:
        w = csv.writer(f)
        for i in xrange(args.batch_size):
            w.writerow(decode((x, y_1, y_2), i, idx_word_map) + [str(real_label[i]), str(pred_label[i])])


if __name__ == '__main__':

    import json
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    # rnn_dim 1000 before
    parser.add_argument('--rnn_dim', type=int, default=512, help='dimension of recurrent states and embeddings')
    parser.add_argument('--rlayers', type=int, default=1, help='number of hidden layers for RNNs')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batches')
    parser.add_argument('--input_size', type=int, default=300, help='size of x input, embed size')
    parser.add_argument('--tgt_vocab_size', type=int, default=0, help='source vocabulary size')
    parser.add_argument('--warm_start_lr', type=float, default=None,
                        help='train for warm_start_iters using this lr before switching to larger lr')
    parser.add_argument('--warm_start_iters', type=int, default=100, help='see warm_start_lr')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--lr_decay_after', type=int, default=3, help='epoch after which to decay')
    parser.add_argument('--lr_decay_threshold', type=float, default=0.01,
                        help='begin decaying learning rate if diff of prev 2 validation costs less than thresohld')
    parser.add_argument('--max_lr_decays', type=int, default=8, help='maximum number of times to decay learning rate')
    parser.add_argument('--max_norm', type=float, default=5.0, help='norm at which to rescale gradients')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout (fraction of units randomly dropped on non-recurrent connections)')
    parser.add_argument('--recdrop', action='store_true',
                        help='use dropout on recurrent updates if True, use stocdrop if False')  # we don't use this
    parser.add_argument('--stocdrop', type=float, default=0.0,
                        help='use to set droprate for stocdrop or recdrop')  # won't use this either
    parser.add_argument('--bidir', action='store_true', help='whether to use bidirectional (GRU)')  # bidir by default
    parser.add_argument('--src_steps', type=int, default=65, help='source sequence length')
    parser.add_argument('--tgt_steps', type=int, default=20, help='target sequence length')
    parser.add_argument('--label_size', type=int, default=2, help='number of labels that we need to predict')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--print_every', type=int, default=1, help='how often to print cost')
    parser.add_argument('--optimizer', type=str, default='adam', choices=optimizers)
    parser.add_argument('--reverse', action='store_true', help='reverse source input sentence')
    parser.add_argument('--expdir', type=str, default='sandbox', help='experiment directory to save files to')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='resume starting at this epoch (expdir must have saved models, and this will also load previous opts.json for configuration)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--embed', type=str, default='w2v',
                        choices=['w2v', 'glove50', 'glove100', 'glove200', 'glove300'],
                        help='which embedding to load in')
    parser.add_argument('--mode', type=str, default='sq2sq', choices=['sq2sq', 'att', 'gen'],
                        help='which embedding to load in')
    parser.add_argument('--attention', action='store_true',
                        help='whether we want to turn on attention')
    parser.add_argument('--pretrain', action='store_true', help='whether we want to pretrain on ROCStories')
    parser.add_argument('--vector_mode', type=str, default='tgt1', choices=['tgt1', 'concat', 'sub'])
    parser.add_argument('--save_err', action='store_true',
                        help='will save test/validation result to a csv file in expdir')
    parser.add_argument('--load_model', type=str, default='',
                        help='load a specific epoch and run only once to inspect the model, only put directory path, will automatically load best epoch.')

    args = parser.parse_args()

    # build the model according to params
    if not args.load_model:
        logger, opts = setup_exp(args)
        logger.info(args)
    else:
        logger = get_logger(args.expdir)
        with open(pjoin(args.load_model, 'opts.json'), 'r') as fin:
            loaded_opts = json.load(fin)
            for k in loaded_opts:
                if k not in ['expdir', 'load_model', 'seed']:
                    setattr(args, k, loaded_opts[k])
        logger.info(args)

    logger.info('loading data...')

    word_idx_map, idx_word_map = load_vocab(VOCAB_PATH)
    vocab_size = len(idx_word_map)

    loader = StoryLoader(STORY_DATA_PATH,
                         batch_size=args.batch_size, src_seq_len=65,
                         tgt_seq_len=20, mode='merged')

    if args.embed == 'w2v':
        embedding = loader.get_w2v_embed().astype('float32')
    else:
        raise NotImplementedError

    args.tgt_vocab_size = embedding.shape[0] # must obtain it here

    start_epoch = 0

    logger.info('building model...')

    story_model = StoryModelSeq2Seq(args, embedding, attention=args.attention)

    logger.info('# params: %d' % story_model.nparams)

    if args.save_err:
        import sys

        logger.info('inspecting model: ' + args.load_model)
        with open(pjoin(args.load_model, 'costs.pkl'), 'rb') as fin:
            mean_valid_costs, all_train_costs, final_test_cost, \
            all_train_accu, mean_valid_accus, final_test_accu, decay_epochs = pickle.load(fin)

        epoch_min = np.argmin(np.asarray(mean_valid_costs))
        load_model_params(story_model, pjoin(args.load_model, 'model_epoch%d.pk' % epoch_min))

        loc = pjoin(args.load_model, 'error_analysis.csv')

        for k in xrange(loader.val_num_batches):
            x, (y, y_2), real_label = loader.get_batch('val', k)
            # [src_sent, tgt_sent, labels, theano.In(pdrop, value=0.0)]
            ret = story_model.test(x, y, y_2, real_label, 0.0)
            cost, preds = ret[0:2]

            save_result(loc, x, y, y_2, real_label, preds, idx_word_map)

        for k in xrange(loader.test_num_batches):
            x, (y, y_2), real_label = loader.get_batch('test', k)
            ret = story_model.test(x, y, y_2, real_label, 0.0)
            cost, preds = ret[0:2]

            save_result(loc, x, y, y_2, real_label, preds, idx_word_map)

        sys.exit(0)

    lr = args.lr

    pretrain_time = 0
    if args.pretrain:
        # we pretrain here
        bigtic = time.time()
        all_train_costs = list()
        mean_train_costs = list()
        decayed = 0
        decay_epochs = list()
        expcost = None
        for epoch in xrange(start_epoch, args.epochs):
            # still want to decay learning rate
            if epoch >= args.lr_decay_after:
                if len(mean_train_costs) > 1 and mean_train_costs[-2] - mean_train_costs[-1] < args.lr_decay_threshold:
                    decayed += 1
                    if mean_train_costs[-2] - mean_train_costs[-1] < 0:
                        logger.info('changing parameters back to epoch %d' % (epoch - 1))
                        load_model_params(story_model, pjoin(args.expdir, 'model_epoch%d.pk' % (epoch - 1)))
                    if decayed > args.max_lr_decays:
                        break
                    logger.info('annealing at epoch %d from %f to %f' % (epoch, lr, lr * args.lr_decay))
                    decay_epochs.append(epoch)
                    # if we did worse this validation epoch load last epoch's parameters
                    lr = lr * args.lr_decay

            it = 0
            curr_train_cost = []

            for k in xrange(loader.pre_train_num_batches):
                tic = time.time()
                it += 1
                x, y = loader.get_pretrain_batch(k)
                # src_sent, tgt_sent, pdrop, lr, max_norm
                ret = story_model.pretrain(x, y, args.dropout, lr, args.max_norm)
                cost, grad_norm, param_norm = ret[0:3]

                norm_ratio = grad_norm / param_norm
                all_train_costs.append(cost)
                curr_train_cost.append(cost)

                if not expcost:
                    expcost = cost
                else:
                    expcost = 0.01 * cost + 0.99 * expcost
                toc = time.time()

                if (it + 1) % args.print_every == 0:
                    logger.info('pretrain epoch %d, iter %d, cost %f, expcost %f, batch time %f, grad/param norm %f' % \
                                (epoch + 1, it, cost, expcost, toc - tic, norm_ratio))

            mean_train_costs.append(np.mean(curr_train_cost))
            logger.info('pretrain epoch %d, mean training accuracy: %f' % (epoch + 1, np.mean(curr_train_cost)))
        bigtoc = time.time()
        pretrain_time = bigtoc - bigtic

    all_train_costs = list()  # at every batch
    all_valid_costs = list()  # at every batch
    mean_valid_costs = list()  # at end of every epoch
    epoch_cost_dict = dict()

    all_train_accu = list()
    mean_train_accu = list()
    all_valid_accu = list()
    mean_valid_accus = list()
    epoch_accu_dict = dict()

    decayed = 0
    decay_epochs = list()

    expcost = None

    bigtic = time.time()
    for epoch in xrange(start_epoch, args.epochs):

        if epoch >= args.lr_decay_after:
            if len(mean_valid_costs) > 1 and mean_valid_costs[-2] - mean_valid_costs[-1] < args.lr_decay_threshold:
                decayed += 1
                if mean_valid_costs[-2] - mean_valid_costs[-1] < 0:
                    logger.info('changing parameters back to epoch %d' % (epoch - 1))
                    load_model_params(story_model, pjoin(args.expdir, 'model_epoch%d.pk' % (epoch - 1)))
                if decayed > args.max_lr_decays:
                    break
                logger.info('annealing at epoch %d from %f to %f' % (epoch, lr, lr * args.lr_decay))
                decay_epochs.append(epoch)
                # if we did worse this validation epoch load last epoch's parameters
                lr = lr * args.lr_decay

        it = 0
        curr_train_accu = []  # training accuracy for the current batch
        for k in xrange(loader.train_num_batches):
            tic = time.time()
            it += 1
            x, (y, y_2), real_label = loader.get_batch('train', k)
            # [src_sent, tgt_sent, labels, pdrop, lr, max_norm]
            ret = story_model.train(x, y, y_2, real_label, args.dropout, lr, args.max_norm)
            cost, grad_norm, param_norm, train_preds = ret[0:4]

            train_accuracy = np.mean(real_label == train_preds)

            norm_ratio = grad_norm / param_norm
            all_train_costs.append(cost)
            all_train_accu.append(train_accuracy)
            curr_train_accu.append(train_accuracy)

            if not expcost:
                expcost = cost
            else:
                expcost = 0.01 * cost + 0.99 * expcost
            toc = time.time()

            if (it + 1) % args.print_every == 0:
                logger.info('epoch %d, iter %d, cost %f, expcost %f, batch time %f, grad/param norm %f, accuracy %f' % \
                            (epoch + 1, it, cost, expcost, toc - tic, norm_ratio, train_accuracy))

        mean_train_accu.append(np.mean(curr_train_accu))

        logger.info('epoch %d, mean training accuracy: %f' % (epoch + 1, np.mean(curr_train_accu)))
        # run on validation
        valid_costs = []
        valid_accu = []
        for k in xrange(loader.val_num_batches):
            x, (y, y_2), real_label = loader.get_batch('val', k)
            # [src_sent, tgt_sent, labels, theano.In(pdrop, value=0.0)]
            ret = story_model.test(x, y, y_2, real_label, 0.0)
            cost, preds = ret[0:2]

            val_accuracy = np.mean(real_label == preds)

            valid_accu.append(val_accuracy)
            all_valid_costs.append(cost)
            valid_costs.append(cost)  # local

        mean_valid_cost = sum(valid_costs) / float(len(valid_costs))
        mean_valid_accu = sum(valid_accu) / float(len(valid_accu))
        mean_valid_costs.append(mean_valid_cost)
        mean_valid_accus.append(mean_valid_accu)
        logger.info('validation cost: %f, accuracy: %f' % (mean_valid_cost, mean_valid_accu))
        epoch_cost_dict[epoch] = mean_valid_cost
        epoch_accu_dict[epoch] = mean_valid_accu

        logger.info('saving model')
        save_model_params(story_model, pjoin(args.expdir, 'model_epoch%d.pk' % epoch))

    best_valid_accu_epoch = sorted(epoch_accu_dict, key=epoch_accu_dict.get)[0]

    best_valid_epoch = sorted(epoch_cost_dict, key=epoch_cost_dict.get)[0]
    # restore model at best validation epoch
    load_model_params(story_model, pjoin(args.expdir, 'model_epoch%d.pk' % best_valid_epoch))

    test_accu = list()
    test_costs = list()
    for k in xrange(loader.test_num_batches):
        x, (y, y_2), real_label = loader.get_batch('test', k)
        ret = story_model.test(x, y, y_2, real_label, 0.0)
        cost, preds = ret[0:2]

        accu = np.mean(real_label == preds)

        test_accu.append(accu)
        test_costs.append(cost)

    final_test_cost = sum(test_costs) / float(len(test_costs))
    final_test_accu = sum(test_accu) / float(len(test_accu))

    if args.pretrain:
        logger.info('final pretraining cost: %f' % min(mean_train_costs))
        logger.info('total pretraining time: %f m' % (pretrain_time / 60.))
        logger.info('')

    logger.info('best training accuracy: %f' % max(mean_train_accu))
    logger.info('best validation accuracy: %f' % max(mean_valid_accus))
    logger.info('final test accuracy: %f' % final_test_accu)
    logger.info('best validation cost: %f' % epoch_cost_dict[best_valid_epoch])
    logger.info('final test cost using epoch %d parameters: %f' % (best_valid_epoch, final_test_cost))

    bigtoc = time.time()
    logger.info('')
    logger.info('total training time: %f m' % ((bigtoc - bigtic) / 60.))

    with open(pjoin(args.expdir, 'costs.pkl'), 'wb') as f:
        pickle.dump((mean_valid_costs, all_train_costs, final_test_cost,
                     all_train_accu, mean_valid_accus, final_test_accu, decay_epochs), f)
