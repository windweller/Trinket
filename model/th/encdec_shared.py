import numpy as np
import theano
from theano import tensor as T
from ug_utils import floatX, Dropout
from rnn import (RNN, SequenceLogisticRegression, GRULayer, GRULayerAttention,
        LayerWrapper, seq_cat_crossent, Downscale)
from opt import get_opt_fn
from ug_utils import glorot_init, norm_init, uniform_init, get_sequence_dropout_mask

def reverse_sent(ss, sm):
    rss = np.array(ss)
    lens = np.sum(sm, axis=1)
    for k in xrange(rss.shape[0]):
        rss[k, 0:lens[k]] = rss[k, lens[k]-1::-1]
    return rss

def clip_lengths(sents, mask, clip_length):
    max_length = np.max(np.sum(mask, axis=1))
    if max_length > clip_length:
        sents = sents[:, :clip_length]
        mask = mask[:, :clip_length]
    else:
        pass
    return sents, mask

def pyrpad(ss, sm, depth, pyramid):
    if not pyramid:
        return ss, sm
    align = pow(2, depth - 1)
    batch_size, seq_len = ss.shape
    padlen = (align - seq_len) % align
    if padlen == 0:
        return ss, sm
    padding = np.zeros([batch_size, padlen])
    return np.concatenate((ss, padding), axis=1), np.concatenate((sm, padding), axis=1)

def extract_unmasked(hs, mask):
    # for each batch, extract indices where mask==1 and put into output matrix
    lens = T.sum(mask, axis=0)
    maxlen = T.max(lens)
    #maxlen = theano.printing.Print(maxlen)(maxlen)  # should be # words
    out = T.zeros((hs.shape[1], maxlen, hs.shape[2])) - 1
    def extract_step(h, m, k, out):
        inds = m.nonzero()[0]
        out = T.set_subtensor(out[k, 0:inds.shape[0], :], h[inds, :])
        return k + 1, out[k]
    pair, updates = theano.scan(extract_step, sequences=[hs.dimshuffle((1, 0, 2)), mask.dimshuffle((1, 0))], outputs_info=[0, None], non_sequences=out)
    out = pair[1]
    out = out.dimshuffle((1, 0, 2))
    #out = theano.printing.Print(out)(out)
    return out

class RNNDecoder(RNN):

    def __init__(self, rep, y, mask, L_dec, pdrop, args):
        self.h0s = rep
        outputs_info = self.h0s
        rlayers = list()
        self.subset = L_dec[y.flatten()]
        inp = self.subset.reshape((y.shape[0], y.shape[1], L_dec.shape[1]))
        seqmask = get_sequence_dropout_mask((y.shape[0], y.shape[1], L_dec.shape[1]), pdrop)
        # exclude last prediction
        inplayer = GRULayer(inp[:-1].astype(floatX), mask[:-1], seqmask[:-1], args.rnn_dim,
                outputs_info[0], args, suffix='dec0')
        rlayers.append(inplayer)
        for k in xrange(1, args.rlayers):
            seqmask = get_sequence_dropout_mask((y.shape[0], y.shape[1], args.rnn_dim), pdrop)
            rlayer = GRULayer(Dropout(rlayers[-1].out, pdrop).out, mask[:-1], seqmask[:-1],
                    args.rnn_dim, outputs_info[k], args, suffix='dec%d' % k)
            rlayers.append(rlayer)
        olayer = SequenceLogisticRegression(Dropout(rlayers[-1].out, pdrop).out, args.rnn_dim,
                args.tgt_vocab_size)
        cost = seq_cat_crossent(olayer.out, y[1:], mask[1:], normalize=False)
        super(RNNDecoder, self).__init__(rlayers, olayer, cost)

class RNNDecoderAttention(RNN):

    def __init__(self, encoder, y, mask, L_dec, pdrop, args):
        self.hs = encoder.hs
        # NOTE just use this so only last layer uses attention
        def layer_init(attention):
            if not attention:
                return GRULayer
            else:
                return lambda *largs, **kwargs: GRULayerAttention(self.hs, *largs, **kwargs)
        # initial states
        outputs_info = [T.zeros_like(self.hs[0]) for k in xrange(len(encoder.routs))]
        rlayers = list()
        self.subset = L_dec[y.flatten()]
        inp = self.subset.reshape((y.shape[0], y.shape[1], L_dec.shape[1]))
        attention = args.rlayers == 1
        # exclude last prediction
        seqmask = get_sequence_dropout_mask((y.shape[0], y.shape[1], L_dec.shape[1]), pdrop)
        inplayer = layer_init(attention)(inp[:-1].astype(floatX), mask[:-1], seqmask[:-1], args.rnn_dim,
                outputs_info[0], args, suffix='dec0')
        rlayers.append(inplayer)
        for k in xrange(1, args.rlayers):
            attention = (args.rlayers == k + 1)
            seqmask = get_sequence_dropout_mask((y.shape[0], y.shape[1], args.rnn_dim), pdrop)
            rlayer = layer_init(attention)(Dropout(rlayers[-1].out, pdrop).out, mask[:-1],
                    seqmask[:-1], args.rnn_dim, outputs_info[k], args, suffix='dec%d' % k)
            rlayers.append(rlayer)
        olayer = SequenceLogisticRegression(Dropout(rlayers[-1].out, pdrop).out, args.rnn_dim,
                args.tgt_vocab_size)
        cost = seq_cat_crossent(olayer.out, y[1:], mask[1:], normalize=False)
        super(RNNDecoderAttention, self).__init__(rlayers, olayer, cost)

class RNNEncoder(RNN):

    def __init__(self, x, mask, space_mask, L_enc, pdrop, args, suffix_prefix='enc', backwards=False):
        # NOTE shape[1] is batch size since shape[0] is seq length
        outputs_info = [T.zeros((x.shape[1], args.rnn_dim)).astype(floatX)]
        rlayers = list()
        self.subset = L_enc[x.flatten()]
        inp = self.subset.reshape((x.shape[0], x.shape[1], L_enc.shape[1]))
        seqmask = get_sequence_dropout_mask((x.shape[0], x.shape[1], L_enc.shape[1]), pdrop)
        inplayer = GRULayer(inp.astype(floatX), mask, seqmask, args.input_size, outputs_info,
                args, suffix='%s0' % suffix_prefix, backwards=backwards)
        rlayers.append(inplayer)
        for k in xrange(1, args.rlayers):
            inp = rlayers[-1].out
            seqmask = get_sequence_dropout_mask((x.shape[0], x.shape[1], args.rnn_dim), pdrop)
            rlayer = GRULayer(Dropout(inp, pdrop).out, mask, seqmask, args.rnn_dim,
                    outputs_info, args, suffix='%s%d' % (suffix_prefix, k), backwards=backwards)
            rlayers.append(rlayer)

        # should extract final outputs according to mask, note we
        # don't know seq length or current batch size at graph construction time
        # NOTE this would be used for initial hidden states in decoder in standard seq2seq but currently unused
        lens = T.sum(mask, axis=0)
        # will extract A[lens[k], k, :] for k in [0, batch size)
        self.routs = list()
        for rlayer in rlayers:
            # get the last time steps (what's this doing with batch???)
            rout = rlayer.out[args.src_steps - 1, theano.tensor.arange(x.shape[1]), :].astype(floatX)
            self.routs.append(rout)
        self.hs = rlayers[-1].out  # for attention

        olayer = LayerWrapper(self.routs)
        super(RNNEncoder, self).__init__(rlayers, olayer)

class BiPyrRNNEncoder(RNN):
    def __init__(self, x, xr, mask, L_enc, pdrop, args):
        # NOTE shape[1] is batch size since shape[0] is seq length
        outputs_info = [T.zeros((x.shape[1], args.rnn_dim)).astype(floatX)]
        flayers = list()
        blayers = list()
        fsubset = L_enc[x.flatten()]
        bsubset = L_enc[xr.flatten()]
        finp = fsubset.reshape((x.shape[0], x.shape[1], L_enc.shape[1]))
        binp = bsubset.reshape((x.shape[0], x.shape[1], L_enc.shape[1]))
        fseqmask = get_sequence_dropout_mask((x.shape[0], x.shape[1], L_enc.shape[1]), pdrop)
        bseqmask = get_sequence_dropout_mask((x.shape[0], x.shape[1], L_enc.shape[1]), pdrop)
        finplayer = GRULayer(finp.astype(floatX), mask, fseqmask, args.rnn_dim, outputs_info,
                args, suffix='fenc0')
        binplayer = GRULayer(binp.astype(floatX), mask, bseqmask, args.rnn_dim, outputs_info,
                args, suffix='benc0', backwards=True)
        flayers.append(finplayer)
        blayers.append(binplayer)
        self.routs = list()  # unlike RNNEncoder, contains hs, not just final h
        self.routs.append(finplayer.out + binplayer.out)
        downs = []
        for k in xrange(1, args.rlayers):
            # concatenate consecutive steps in the sequence (which are downscaled to half from the previous layer)
            d = Downscale(self.routs[-1], args.rnn_dim, suffix='ds%d' % k)
            downs.append(d)
            inp = d.out
            twocols = mask.T.reshape([-1, 2])
            mask = T.or_(twocols[:, 0], twocols[:, 1]).reshape([mask.shape[1], -1]).T

            fseqmask = get_sequence_dropout_mask((inp.shape[0], inp.shape[1], args.rnn_dim), pdrop)
            bseqmask = get_sequence_dropout_mask((inp.shape[0], inp.shape[1], args.rnn_dim), pdrop)
            flayer = GRULayer(Dropout(inp, pdrop).out, mask, fseqmask, args.rnn_dim, outputs_info, args, suffix='fenc%d' % k)
            blayer = GRULayer(Dropout(inp, pdrop).out, mask, bseqmask, args.rnn_dim, outputs_info, args, suffix='benc%d' % k, backwards=True)
            self.routs.append(flayer.out + blayer.out)
            flayers.append(flayer)
            blayers.append(blayer)
        self.hs = self.routs[-1]  # for attention
        olayer = LayerWrapper(self.routs)
        rlayers = flayers + blayers  # NOTE careful not to assume rlayers = # layers in all cases

        # undo the temporary hack
        super(BiPyrRNNEncoder, self).__init__(rlayers, olayer, downscales=downs)

class BiRNNEncoder(RNN):

    def __init__(self, x, xr, mask, space_mask, L_enc, pdrop, args):
        # NOTE shape[1] is batch size since shape[0] is seq length
        outputs_info = [T.zeros((x.shape[1], args.rnn_dim)).astype(floatX)]
        flayers = list()
        blayers = list()

        finp = L_enc[x]
        binp = L_enc[xr]

        # fsubset = L_enc[x.flatten()]
        # bsubset = L_enc[xr.flatten()]
        # finp = fsubset.reshape((x.shape[0], x.shape[1], L_enc.shape[1]))
        # binp = bsubset.reshape((x.shape[0], x.shape[1], L_enc.shape[1]))

        fseqmask = get_sequence_dropout_mask((x.shape[0], x.shape[1], L_enc.shape[1]), pdrop)
        bseqmask = get_sequence_dropout_mask((x.shape[0], x.shape[1], L_enc.shape[1]), pdrop)
        finplayer = GRULayer(finp.astype(floatX), mask, fseqmask, args.input_size, outputs_info,
                args, suffix='fenc0')
        binplayer = GRULayer(binp.astype(floatX), mask, bseqmask, args.input_size, outputs_info,
                args, suffix='benc0', backwards=True)
        flayers.append(finplayer)
        blayers.append(binplayer)
        self.routs = list()  # unlike RNNEncoder, contains hs, not just final h
        self.routs.append(finplayer.out + binplayer.out)
        for k in xrange(1, args.rlayers):
            inp = self.routs[-1]
            fseqmask = get_sequence_dropout_mask((inp.shape[0], inp.shape[1], args.rnn_dim), pdrop)
            bseqmask = get_sequence_dropout_mask((inp.shape[0], inp.shape[1], args.rnn_dim), pdrop)
            flayer = GRULayer(Dropout(inp, pdrop).out, mask, fseqmask, args.rnn_dim, outputs_info, args, suffix='fenc%d' % k)
            blayer = GRULayer(Dropout(inp, pdrop).out, mask, bseqmask, args.rnn_dim, outputs_info, args, suffix='benc%d' % k, backwards=True)
            self.routs.append(flayer.out + blayer.out)
            flayers.append(flayer)
            blayers.append(blayer)
        self.hs = self.routs[-1]  # for attention
        olayer = LayerWrapper(self.routs)
        rlayers = flayers + blayers  # NOTE careful not to assume rlayers = # layers in all cases
        super(BiRNNEncoder, self).__init__(rlayers, olayer)

class EncoderDecoder(object):

    # subset_grad determines whether want to use subset updates (faster but
    # currently can only use sgd) or full updates on embeddings
    def __init__(self, args, params=None, attention=False, bidir=False, subset_grad=True, pyramid=False):
        self.rnn_dim = args.rnn_dim
        self.rlayers = args.rlayers
        self.attention = attention

        lr = T.scalar(dtype=floatX)
        pdrop = T.scalar(dtype=floatX)
        max_norm = T.scalar(dtype=floatX)

        # initialize input tensors

        src_sent = T.imatrix('src_sent')
        rev_src_sent = T.imatrix('rev_src_sent')
        src_mask = T.bmatrix('src_mask')
        tgt_sent = T.imatrix('tgt_sent')
        tgt_mask = T.bmatrix('tgt_mask')
        space_mask = T.bmatrix('space_mask')

        # build up model
        # https://groups.google.com/forum/#!topic/torch7/-NBrFw8Q6_s
        # NOTE can't use one-hot here because huge matrix multiply
        self.L_enc = theano.shared(uniform_init(args.src_vocab_size, args.rnn_dim, scale=0.1),
                'L_enc', borrow=True)
        self.L_dec = theano.shared(uniform_init(args.tgt_vocab_size, args.rnn_dim, scale=0.1),
                'L_dec', borrow=True)
        enc_input = src_sent if not args.reverse else rev_src_sent
        if bidir:
            print('Using bidirectional encoder')
            self.encoder = BiRNNEncoder(src_sent.T, rev_src_sent.T, src_mask.T, space_mask.T, self.L_enc, pdrop, args)
        elif pyramid:
            print('Using pyramid encoder')
            self.encoder = BiPyrRNNEncoder(src_sent.T, rev_src_sent.T, src_mask.T, self.L_enc, pdrop, args)
        else:
            self.encoder = RNNEncoder(enc_input.T, src_mask.T, space_mask.T, self.L_enc, pdrop, args)
        if attention:
            self.decoder = RNNDecoderAttention(self.encoder, tgt_sent.T, tgt_mask.T,
                    self.L_dec, pdrop, args)
            hs = self.decoder.hs
        else:
            self.decoder = RNNDecoder(self.encoder.out, tgt_sent.T, tgt_mask.T,
                    self.L_dec, pdrop, args)

        # cost, parameters, grads, updates

        self.cost = self.decoder.cost
        self.params = self.encoder.params + self.decoder.params + [self.L_enc, self.L_dec]
        if subset_grad:  # for speed
            self.grad_params = self.encoder.params + self.decoder.params + [self.encoder.subset, self.decoder.subset]
            self.updates, self.grad_norm, self.param_norm = get_opt_fn(args.optimizer)(self.cost, self.grad_params, lr, max_norm=max_norm)
            # instead of updating L_enc and L_dec only want to update the embeddings indexed, so use inc_subtensor/set_subtensor
            # http://deeplearning.net/software/theano/tutorial/faq_tutorial.html
            self.updates[-2] = (self.L_enc, T.set_subtensor(self.updates[-2][0], self.updates[-2][1]))
            self.updates[-1] = (self.L_dec, T.set_subtensor(self.updates[-1][0], self.updates[-1][1]))
        else:
            self.grad_params = self.params
            self.updates, self.grad_norm, self.param_norm = get_opt_fn(args.optimizer)(self.cost, self.grad_params, lr, max_norm=max_norm)

        self.nparams = np.sum([np.prod(p.shape.eval()) for p in self.params])

        # functions

        self.train = theano.function(
            inputs=[src_sent, src_mask, rev_src_sent, tgt_sent, tgt_mask, space_mask,
                pdrop, lr, max_norm],
            outputs=[self.cost, self.grad_norm, self.param_norm],
            updates = self.updates,
            on_unused_input='warn',
            allow_input_downcast=True
        )
        self.test = theano.function(
            inputs=[src_sent, src_mask, rev_src_sent, tgt_sent, tgt_mask, space_mask, theano.In(pdrop, value=0.0)],
            outputs=self.cost,
            updates=None,
            on_unused_input='warn'
        )
        outputs=self.encoder.out
        if attention:
            outputs = self.encoder.out + [hs]
        self.encode = theano.function(
            inputs=[src_sent, rev_src_sent, src_mask, space_mask, theano.In(pdrop, value=0.0)],
            outputs=outputs,
            on_unused_input='warn',
            updates=None
        )

        # function for decoding step by step

        i_t = T.ivector()
        x_t = self.L_dec[i_t, :]
        h_ps = list()  # previous
        for k in xrange(args.rlayers):
            h_ps.append(T.matrix())
        h_ts = list()
        dmask = T.ones_like(h_ps[0]).astype(floatX)
        if attention and args.rlayers == 1:
            h_t, _ = self.decoder.rlayers[0]._step(x_t, dmask, h_ps[0], hs)
        else:
            h_t = self.decoder.rlayers[0]._step(x_t, dmask, h_ps[0])
        h_ts.append(h_t)
        # NOTE no more dropout nodes here
        for k in xrange(1, args.rlayers):
            if attention and args.rlayers == k + 1:
                h_t, align = self.decoder.rlayers[k]._step(h_t, dmask, h_ps[k], hs)
            else:
                h_t = self.decoder.rlayers[k]._step(h_t, dmask, h_ps[k])
            h_ts.append(h_t)
        E_t = T.dot(h_t, self.decoder.olayer.W) + self.decoder.olayer.b
        E_t = T.exp(E_t - T.max(E_t, axis=1, keepdims=True))
        p_t = E_t / E_t.sum(axis=1, keepdims=True)
        inputs=[i_t] + h_ps
        outputs=[p_t] + h_ts
        if attention:
            inputs = inputs + [hs]
            outputs = outputs + [align]
        self.decode_step = theano.function(
            inputs=inputs,
            outputs=outputs,
            updates=None
        )
