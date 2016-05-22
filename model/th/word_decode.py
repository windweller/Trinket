import time
import numpy as np
from ug_utils import floatX
from run_utils import get_logger, restore_model_and_opts
from word_fuel import load_data
from word_encdec import RNNEncoder, RNNDecoder, EncoderDecoder, reverse_sent
from ug_cfg import EOS_IND, WORD_MAX_SEQ_LEN, BLEU_SCORER, SRC_VALID_FILE,\
        TGT_VALID_FILE, SRC_TEST_FILE, TGT_TEST_FILE, WORD_SORT_K_BATCHES,\
        TGT_VOCAB_SIZE
from os.path import join as pjoin

'''
given translation model runs decoding
'''

# FIXME still seems broken, at least on mt tests
def decode_sent(model, prev_states, bdict, beam_size=1, beta=1.0):
    prev_hyps = np.array([EOS_IND], dtype=np.int32).reshape((1,1))
    prev_probs = np.log(np.array([1.0]).reshape(1, 1))
    finished_hyps = list()
    finished_probs = list()
    nlayers = len(prev_states)
    next_states = [np.zeros((beam_size, prev_states[0].shape[1]), dtype=floatX) for k in xrange(nlayers)]
    # NOTE decoded sentence length capped at training length
    for l in xrange(WORD_MAX_SEQ_LEN):
        # pass in input and states from last time step
        ret = model.decode_step(*([prev_hyps[:, -1:].flatten()] + prev_states))
        # should be of shape [beam_size (or 1 for 1st iter), tgt_vocab_size]
        logprobs = np.log(ret[0])
        states = ret[1:]
        next_hyps = list()
        curr_probs = (prev_probs.reshape((-1, 1)) + logprobs).flatten()
        # want top from high to low prob hence reverse
        max_inds = np.argpartition(curr_probs, -2*beam_size)[-2*beam_size:].flatten()[::-1]
        max_probs = curr_probs[max_inds]
        prev_probs = np.zeros(beam_size)
        non_eos = 0
        for j in xrange(max_inds.shape[0]):
            hyp_ind = max_inds[j] / TGT_VOCAB_SIZE
            vocab_ind = max_inds[j] % TGT_VOCAB_SIZE
            if vocab_ind == EOS_IND:
                finished_hyps.append(list(prev_hyps[hyp_ind]) + [vocab_ind])
                finished_probs.append(max_probs[j])
                continue
            for k in xrange(nlayers):
                next_states[k][non_eos] = states[k][hyp_ind]
            prev_probs[non_eos] = max_probs[j]
            next_hyps.append(np.array(list(prev_hyps[hyp_ind]) + [vocab_ind], dtype=np.int32))
            non_eos += 1
            if non_eos >= beam_size:
                break
        if len(next_hyps) == 0:
            break
        prev_hyps = np.concatenate(next_hyps, axis=0).reshape((len(next_hyps), -1))
        prev_states = next_states
    if len(finished_hyps) == 0:
        finished_hyps = [list(prev_hyps[k]) for k in xrange(prev_hyps.shape[0])]
        finished_probs = list(prev_probs)
    # now sort
    hyps_and_probs = sorted(zip(finished_hyps, finished_probs), key=lambda pair: pair[1] + beta * len(pair[0]))
    best_hyp = hyps_and_probs[-1][0]
    # remove eos at the start
    best_hyp = best_hyp[1:]
    # handle repeats
    if len(best_hyp) > 4:  # only keep 1st of single repeats
        for k in xrange(4, len(best_hyp)):
            if len(set(best_hyp[k-4:k])) == 1:
                best_hyp = best_hyp[:k-3]
    if len(best_hyp) > 6:  # only keep 1st of pair repeats
        for k in xrange(6, len(best_hyp)):
            if len(set(best_hyp[k-6:k])) == 2:
                best_hyp = best_hyp[:k-4]
    best_str = ' '.join([bdict[i] for i in best_hyp])
    best_str = best_str.split('</s>')[0]
    best_str = best_str.strip()
    return best_str, hyps_and_probs[-1][1]

if __name__ == '__main__':
    import json
    import argparse
    import logging
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=str, help='experiment directory')
    parser.add_argument('model', type=str, help='model file to use')
    parser.add_argument('outfile', type=str, help='file to write decoder output setnences to')
    parser.add_argument('--beam', type=int, default=12, help='beam size to use')
    parser.add_argument('--test', action='store_true', help='evaluate on test set instead of validation set')
    parser.add_argument('--beta', type=float, default=1.0, help='insertion bonus')
    args = parser.parse_args()
    args.reverse = True  # just in case

    logger = get_logger(args.expdir)
    def model_fn(x): return EncoderDecoder(x)
    model, opts = restore_model_and_opts(model_fn, args, logger)

    # TODO try w/ test as well
    if not args.test:
        eval_stream, src_dict, tgt_dict = load_data(SRC_VALID_FILE, TGT_VALID_FILE,
            opts['batch_size'], WORD_SORT_K_BATCHES, training=False)
    else:
        eval_stream, src_dict, tgt_dict = load_data(SRC_TEST_FILE, TGT_TEST_FILE,
            opts['batch_size'], WORD_SORT_K_BATCHES, training=False)
    tgt_bdict = {v: k for k, v in tgt_dict.iteritems()}

    # finally, run decoding

    decoded_sentences = list()

    decode_costs = []
    lengths = []
    logger.info('starting decoding')
    tic = time.time()
    sent_ind = 0
    for ss, sm, ts, tm in eval_stream.get_epoch_iterator():
        ss, ts = ss.astype(np.int32), ts.astype(np.int32)
        sm, tm = sm.astype(np.bool), tm.astype(np.bool)
        rss = reverse_sent(ss, sm)
        # NOTE batches sometimes different sizes
        cost = model.test(ss, sm, rss, ts, tm)
        length = np.sum(tm[:, 1:])
        decode_costs.append(cost * ss.shape[0])
        lengths.append(length)

        # sample / generate

        for k in xrange(ss.shape[0]):
            # extract single example
            ssk, smk, tsk, tmk = ss[k:k+1, :], sm[k:k+1, :], ts[k:k+1, :], tm[k:k+1, :]
            rssk = reverse_sent(ssk, smk)
            h_t = model.encode(ssk, rssk, smk)
            sent, prob = decode_sent(model, h_t, tgt_bdict, beam_size=args.beam, beta=args.beta)
            print('sent %d, prob %f, translation: %s' % (sent_ind + 1, prob, sent))
            sent_ind = sent_ind + 1
            decoded_sentences.append(sent)

    toc = time.time()
    logger.info('decoding time: %fs' % (toc - tic))
    logger.info('cost: %f' % (sum(decode_costs) / float(sum(lengths))))

    # write to output file

    with open(args.outfile, 'w') as fout:
        fout.write('\n'.join(decoded_sentences))

    # run scoring

    ref_file = TGT_TEST_FILE if args.test else TGT_VALID_FILE
    print('comparing to %s' % ref_file)
    cmd = 'perl %s %s < %s' % (BLEU_SCORER, ref_file, args.outfile)
    logger.info('running "%s"' % cmd)
    subprocess.check_call(cmd, shell=True)
