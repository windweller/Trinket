import theano
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy
import numpy as np
import itertools
from ug_utils import floatX, _linear_params, ortho_init, norm_init

def seq_cat_crossent(pred, targ, mask, normalize=False, return_seq=False):
    # dim 0 is time, dim 1 is batch, dim 2 is category
    pred_flat = pred.reshape(((pred.shape[0] *
                               pred.shape[1]),
                               pred.shape[2]), ndim=2)
    targ_flat = targ.flatten()
    mask_flat = mask.flatten()
    ce = categorical_crossentropy(pred_flat, targ_flat)
    # normalize by batch size and seq length
    ce = ce * mask_flat
    cost = T.sum(ce)
    if normalize:
        # normalize by batch and length
        cost =  cost / T.sum(mask_flat)
    else:
        # just normalize by batch size
        cost = cost / pred.shape[1]
    if not return_seq:
        return cost
    else:
        return cost, ce.reshape((pred.shape[0], pred.shape[1]))

class SequenceLogisticRegression(object):

    # multi-class logistic regression class over sequence

    def __init__(self, inp, n_in, n_out):
        # initialize w/ zeros
        self.W, self.b = _linear_params(n_in, n_out, 'sm')
        E = T.dot(inp, self.W) + self.b
        # time, batch, cat (None just keeps dimension)
        E = T.exp(E - T.max(E , axis=2, keepdims=True))
        pmf = E / T.sum(E, axis=2, keepdims=True)
        self.p_y_given_x = pmf
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.out = self.p_y_given_x
        # parameters of the model
        self.params = [self.W, self.b]

class LayerWrapper(object):

    def __init__(self, out):
        self.out = out
        self.params = []

class RNN(object):

    def __init__(self, rlayers, olayer, cost=None, downscales=None):
        # e.g. GRULayer
        self.rlayers = rlayers
        # e.g. softmax or last state
        self.olayer = olayer
        # e.g. categorical cross-entropy over sequence outputs
        if cost:
            self.cost = cost
        else:
            self.cost = 0
        self.params = list(itertools.chain(*[r.params for r in self.rlayers])) +\
                self.olayer.params
        if downscales:
            for d in downscales:
                self.params = self.params + d.params
        self.out = self.olayer.out

class GRULayer(object):

    def __init__(self, x, mask, seqmask, x_dim, outputs_info, args, suffix='', backwards=False):
        # NOTE if want to stack should equal hdim
        self.xdim = x_dim
        self.hdim = args.rnn_dim
        self.backwards = backwards
        self.recdrop = args.recdrop
        self.stocdrop = args.stocdrop
        # self.var = args.var  # a dev-only parameter for choosing update formulae

        # initialize parameters
        # TODO maybe try initialization here: https://github.com/kyunghyuncho/dl4mt-material/blob/master/session1/nmt.py, helps for memorizing long sequences
        self.W_z, self.b_wz = _linear_params(self.xdim, self.hdim, 'wz%s' % suffix, act=T.nnet.sigmoid)
        self.U_z, self.b_uz = _linear_params(self.hdim, self.hdim, 'uz%s' % suffix, act=T.nnet.sigmoid)
        self.W_r, self.b_wr = _linear_params(self.xdim, self.hdim, 'wr%s' % suffix, act=T.nnet.sigmoid)
        self.U_r, self.b_ur = _linear_params(self.hdim, self.hdim, 'ur%s' % suffix, act=T.nnet.sigmoid)
        self.W_h, self.b_wh = _linear_params(self.xdim, self.hdim, 'wh%s' % suffix)
        self.U_h, self.b_uh = _linear_params(self.hdim, self.hdim, 'uh%s' % suffix)
        self.setup(x, mask, seqmask, outputs_info)

    def flip_nonpadding(self, x, mask):
        #x = theano.printing.Print(x)(x)
        lens = T.sum(mask, axis=0).astype('int32')
        def flip_step(l, k, h):
            # scan iterates batch size times
            tmp = h[:, k, :]
            tmp2 = tmp[:l].copy() # copy here to prevent weird "cycle in graph" theano error
            return k + 1, T.set_subtensor(tmp[:l], tmp2[::-1])
        pair, updates = theano.scan(flip_step, sequences=lens, outputs_info=[0, None], non_sequences=x)
        out = pair[1].dimshuffle((1, 0, 2)) * mask[:, :, None]
        #out = theano.printing.Print(out)(out)
        return out

    def setup(self, x, mask, seqmask, outputs_info):
        self.params = [self.W_z, self.b_wz,
                       self.U_z, self.b_uz,
                       self.W_r, self.b_wr,
                       self.U_r, self.b_ur,
                       self.W_h, self.b_wh,
                       self.U_h, self.b_uh]
        rval, updates = theano.scan(self._step,
                sequences=[x, seqmask], outputs_info=outputs_info)
        # out should be of dim (sequence length, batch size, hidden size)
        self.out = rval * mask[:, :, None]
        # flip everything before the padding (since we want to be able to add forward and backward hidden states)
        if self.backwards:
            self.out = self.flip_nonpadding(self.out, mask)

    def _step(self, x, dmask, prev_h):
        z = T.nnet.sigmoid(T.dot(x, self.W_z) + self.b_wz +\
                           T.dot(prev_h, self.U_z) + self.b_uz)
        r = T.nnet.sigmoid(T.dot(x, self.W_r) + self.b_wr +\
                           T.dot(prev_h, self.U_r) + self.b_ur)
        h = T.tanh(T.dot(x, self.W_h) + self.b_wh +\
                   T.dot(r * prev_h, self.U_h) + self.b_uh)

        if self.recdrop:
            next_h = z * (dmask * h) + (1 - z) * prev_h
        elif self.stocdrop:
            # LSTM: next_c = dmask * (f * prev_c + i * g) + (1-dmask) * prev_c
            #       next_c = f * prev_c + dmask * i * g
            if self.var == 1:
                next_h = dmask * (z * h + (1 - z) * prev_h) + (1 - dmask) * prev_h # old
                # old is very similar to LSTM's current formulation
            else:
                next_h = dmask * z * h + (1 - z) * prev_h  # new
        else:
            next_h = z * h + (1 - z) * prev_h
        return next_h

class GRULayerAttention(GRULayer):

    # assuming this will only be used in decoder hence no backwards option
    def __init__(self, hs, x, mask, seqmask, x_dim, outputs_info, args, suffix=''):
        self.recdrop = args.recdrop
        self.W_concat, self.b_concat = _linear_params(args.rnn_dim * 2, args.rnn_dim, 'concat%s' % suffix)
        self.W_att1, self.b_att1 = _linear_params(args.rnn_dim, args.rnn_dim, 'att1%s' % suffix)
        self.W_att2, self.b_att2 = _linear_params(args.rnn_dim, args.rnn_dim, 'att2%s' % suffix)
        self.hs = hs  # e.g. from encoder
        self.phi_hs = T.tanh(T.dot(self.hs, self.W_att1) + self.b_att1)
        super(GRULayerAttention, self).__init__(x, mask, seqmask, x_dim, outputs_info, args, suffix=suffix)

    def setup(self, x, mask, seqmask, outputs_info):
        self.params = [self.W_z, self.b_wz,
                       self.U_z, self.b_uz,
                       self.W_r, self.b_wr,
                       self.U_r, self.b_ur,
                       self.W_h, self.b_wh,
                       self.U_h, self.b_uh,
                       self.W_concat, self.b_concat,
                       self.W_att1, self.b_att1,
                       self.W_att2, self.b_att2]
        # NOTE this differs from lstm in that we take output from tanh(Wc[h, context]) for softmax
        ret, updates = theano.scan(self._step,
                sequences=[x, seqmask], outputs_info=[outputs_info, None], non_sequences=self.hs)
        self.out = ret[0]
        self.out = mask[:, :, None] * self.out

    def _step(self, x, dmask, prev_h, hs):
        # standard gru update
        z = T.nnet.sigmoid(T.dot(x, self.W_z) + self.b_wz +\
                           T.dot(prev_h, self.U_z) + self.b_uz)
        r = T.nnet.sigmoid(T.dot(x, self.W_r) + self.b_wr +\
                           T.dot(prev_h, self.U_r) + self.b_ur)
        h = T.tanh(T.dot(x, self.W_h) + self.b_wh +\
                   T.dot(r * prev_h, self.U_h) + self.b_uh)
        if self.recdrop:
            h = z * (dmask * h) + (1 - z) * prev_h
        else:
            h = z * h + (1 - z) * prev_h

        # attention mechanism over hs
        # NOTE could move outside, don't think could easily optimize for speed boost
        gamma_h = T.tanh(T.dot(h, self.W_att2) + self.b_att2)
        weights = T.sum(self.phi_hs * gamma_h, axis=2, keepdims=True)  # sum over hidden dimension (dot product of next_h with hs for each batch index), weights seq_length x batch_size x 1
        # exponentiate and normalize
        weights = T.exp(weights - T.max(weights, axis=0, keepdims=True))
        weights = weights / (T.sum(weights, axis=0, keepdims=True) + 1e-6)  # avoid division by 0
        context = T.sum(hs * weights, axis=0)  # sum from 1 to T over hs
        next_h = T.nnet.relu(T.dot(T.concatenate([h, context], axis=1), self.W_concat) + self.b_concat)
        #next_h = T.tanh(T.dot(h + context, self.W_concat))

        return next_h, weights[:, :, 0]

def bn(x, gamma, beta=0.):
    if x.ndim != 2:
        return x
    mean, var = x.mean(axis=0), x.var(axis=0)
    y = T.nnet.bn.batch_normalization(
        inputs=x,
        mean=mean, std=T.sqrt(var + 1e-7),
        gamma=gamma, beta=beta)
    return y

def _slice(x_, n, dim):
    if x_.ndim == 2:
        return x_[:, n*dim:(n+1)*dim]
    return x_[n*dim:(n+1)*dim]

class LSTMLayer(object):

    # follows http://arxiv.org/pdf/1409.2329v5.pdf

    def __init__(self, x, mask, seqmask, x_dim, outputs_info, args, suffix=''):
        # NOTE if want to stack should equal hdim
        self.xdim = x_dim
        self.hdim = args.rnn_dim
        self.recdrop = args.recdrop
        self.stocdrop = args.stocdrop
        self.batch_norm = args.batch_norm
        self.args = args

        if args.ortho:
            W = np.concatenate([norm_init(self.xdim,self.hdim, scale=0.01)]*4,
                                   axis=1)

            U = np.concatenate([ortho_init(self.hdim,self.hdim, scale=0.05)]*4,
                                   axis=1)
            b = np.zeros((4*self.hdim,)).astype(floatX)
            # NOTE need to initialize bias of forget gate of LSTM for best performance!
            # http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
            b[self.hdim:2*self.hdim] = 1.0
            self.W = theano.shared(W, name='W%s' % suffix)
            self.b = theano.shared(b, name='b%s' % suffix)
            self.U = theano.shared(U, name='U%s' % suffix)
        else:
            self.W = _linear_params(self.xdim, 4*self.hdim, 'W%s' % suffix, bias=False)
            b = np.zeros((4*self.hdim,)).astype(floatX)
            b[self.hdim:2*self.hdim] = 1.0
            self.b = theano.shared(b, name='b_W%s' % suffix)
            self.U = _linear_params(self.hdim, 4*self.hdim, 'U%s' % suffix, bias=False)

        self.params = [self.W, self.b, self.U]

        initial_gamma = 0.1

        if self.batch_norm:
            self.gamma_inputs = theano.shared(initial_gamma * np.ones(4*self.hdim,).astype('float32'))
            self.gamma_hiddens = theano.shared(initial_gamma * np.ones(4*self.hdim,).astype('float32'))
            self.gamma_outputs = theano.shared(initial_gamma * np.ones(self.hdim,).astype('float32'))
            self.beta_outputs = theano.shared(np.zeros(self.hdim,).astype('float32'))

            self.params += [self.gamma_inputs, self.gamma_hiddens, self.gamma_outputs, self.beta_outputs]

        if self.args.load_model == '':
            rval, updates = theano.scan(self._step, sequences=[x, seqmask], outputs_info=outputs_info)
        else:
            outputs_info = [outputs_info[0], outputs_info[1]]
            outputs_info += [T.zeros((args.batch_size, args.rnn_dim), dtype=floatX)] * 4
            rval, updates = theano.scan(self._step_monitor, sequences=[x, seqmask], outputs_info=outputs_info)
            # average along batch size, shape (T, D)
            self.activations = [T.mean(rval[i], axis=[1]) for i in range(2, 6)]
            # rval[2], rval[3], rval[4], rval[5]

        # out should be of dim (sequence length, batch size, hidden size)
        self.out = rval[0] * mask[:, :, None]
        self.cell = rval[1] * mask[:, :, None]

    def _step_monitor(self, x, dmask, prev_h, prev_c, i, f, o, g):
        x_ = T.dot(x, self.W)
        h_ = T.dot(prev_h, self.U)

        if self.batch_norm:
            x_ = bn(x_, self.gamma_inputs)
            h_ = bn(h_, self.gamma_hiddens)

        preact = x_ + h_ + self.b

        i = T.nnet.sigmoid(_slice(preact, 0, self.hdim))
        f = T.nnet.sigmoid(_slice(preact, 1, self.hdim))
        o = T.nnet.sigmoid(_slice(preact, 2, self.hdim))
        g = T.tanh(_slice(preact, 3, self.hdim))

        prop_c = f * prev_c + i * g
        if self.recdrop:
            next_c = f * prev_c + i * (dmask * g)
        elif self.stocdrop != 0.0:
            # next_c = (1 - dmask) * (f * prev_c) + dmask * i * g
            #next_c = dmask * prop_c + (1-dmask) * prev_c
            next_c = dmask * prop_c + (1-dmask) * i * g
        else:
            next_c = prop_c

        if not self.batch_norm:
            next_h = o * T.tanh(prop_c)
            #next_h = o * T.tanh(next_c)
        else:
            next_h = o * T.tanh(bn(next_c, self.gamma_outputs, beta=self.beta_outputs))

        return next_h, next_c, i, f, o, g

    def _step(self, x, dmask, prev_h, prev_c):
        x_ = T.dot(x, self.W)
        h_ = T.dot(prev_h, self.U)

        if self.batch_norm:
            x_ = bn(x_, self.gamma_inputs)
            h_ = bn(h_, self.gamma_hiddens)

        preact = x_ + h_ + self.b

        i = T.nnet.sigmoid(_slice(preact, 0, self.hdim))
        f = T.nnet.sigmoid(_slice(preact, 1, self.hdim))
        o = T.nnet.sigmoid(_slice(preact, 2, self.hdim))
        g = T.tanh(_slice(preact, 3, self.hdim))

        prop_c = f * prev_c + i * g
        if self.recdrop:
            next_c = f * prev_c + i * (dmask * g)
        elif self.stocdrop != 0.0:
            # next_c = (1 - dmask) * (f * prev_c) + dmask * i * g
            #next_c = dmask * prop_c + (1-dmask) * prev_c
            next_c = dmask * prop_c + (1-dmask) * i * g
        else:
            next_c = prop_c

        if not self.batch_norm:
            next_h = o * T.tanh(prop_c)
            #next_h = o * T.tanh(next_c)
        else:
            next_h = o * T.tanh(bn(next_c, self.gamma_outputs, beta=self.beta_outputs))

        return next_h, next_c

# Used in PyramidRNN
class Downscale(object):

    def __init__(self, x, dim, suffix=''):
        # NOTE if want to stack should equal hdim

        self.W, self.b = _linear_params(dim * 2, dim, 'ds%s' % suffix)

        # x.shape = [seq_len, batch_size, hdim]
        # x1.shape = [batch_size, seq_len / 2, hdim * 2]
        x1 = x.dimshuffle([1, 0, 2]).reshape([x.shape[1], x.shape[0]/2, x.shape[2] * 2])

        # x2.shape = [batch_size, seq_len / 2, hdim]
        x2 = x1.dot(self.W) + self.b

        # x3.shape = [seq_len / 2, batch_size, hdim]
        x3 = x2.dimshuffle([1, 0, 2])

        self.out = T.tanh(x3)

        self.params = [self.W, self.b]
