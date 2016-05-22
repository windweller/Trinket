import numpy as np
import theano
from ug_utils import floatX
from theano import tensor as T

import loggedfn

# NOTE currently just clipping gradients before computing momentum, etc.

def total_norm(xs):
    norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), xs)))
    return norm

def clip_grads(grads, max_norm=None, max_norm_elemwise=None):
    # TODO may want to not clip embedding grads
    if max_norm:
        grad_norm = total_norm(grads)
        factor = T.maximum(max_norm, grad_norm)
        return [grad * (max_norm/factor) for grad in grads]
    elif max_norm_elemwise:
        grads = [grad.clip(-max_norm_elemwise, max_norm_elemwise) for grad in grads]
    return grads

def sgd(cost, params, lr, max_norm=None, max_norm_elemwise=None):
    grads = clip_grads([T.grad(cost, p) for p in params], max_norm=max_norm, max_norm_elemwise=max_norm_elemwise)
    updates = [
        (param, param - lr * grad) for param, grad in
        zip(params, grads)
    ]
    return updates, total_norm(grads), total_norm(params)

def rmsprop(cost, params, lr, alpha=0.95, eps=1e-8, max_norm=None, max_norm_elemwise=None):
    grads = clip_grads([T.grad(cost, p) for p in params], max_norm=max_norm, max_norm_elemwise=max_norm_elemwise)
    accums = [theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX))
            for p in params]
    updates = [
        (a, alpha * a + (1 - alpha) * T.square(g)) for g, a in zip(grads, accums)
    ]
    # XXX worth fix to assign square(grad) to accum during first iter?
    updates = updates + [
        (p, p - lr * g / (T.sqrt(alpha * a + (1 - alpha) * T.square(g)) + eps)) for p, g, a in zip(params, grads, accums)
    ]
    return updates, total_norm(grads), total_norm(params)

def adagrad(cost, params, lr, eps=1e-8, max_norm=None, max_norm_elemwise=None):
    grads = clip_grads([T.grad(cost, p) for p in params], max_norm=max_norm, max_norm_elemwise=max_norm_elemwise)
    accums = [theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX))
            for p in params]
    updates = [
            (p, p - lr * g / T.sqrt(a + T.square(g) + eps))
            for p, g, a in zip(params, grads, accums)
    ]
    updates += [
        (a, a + T.square(g))
        for a, g in zip(accums, grads)
    ]
    return updates, total_norm(grads), total_norm(params)

# from http://arxiv.org/abs/1412.6980
# and https://gist.github.com/Newmu/acb738767acb4788bac3
# suggested lr 0.001
def adam(cost, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, max_norm=None, max_norm_elemwise=None):
    updates = []
    grads = clip_grads([T.grad(cost, p) for p in params], max_norm=max_norm, max_norm_elemwise=max_norm_elemwise)
    t0 = theano.shared(np.array(0., dtype=floatX))
    t = t0 + 1
    corr1 = (1 - beta1**t)
    corr2 = (1 - beta2**t)
    alpha = lr * T.sqrt(corr2) / corr1
    for p, g in zip(params, grads):
        m = theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX))
        v = theano.shared(value=np.zeros(p.get_value().shape, dtype=floatX))
        m_t = beta1 * m + (1 - beta1) * g
        v_t = beta2 * v + (1 - beta2) * T.square(g)
        p_t = p - alpha * m_t/(T.sqrt(v_t) + eps)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t0, t))
    return updates, total_norm(grads), total_norm(params)

optimizers = ['sgd', 'rmsprop', 'adagrad', 'adam']

def register_and_call(optfn, cost, params, lr, *extra, **kwargs):
    u, gn, pn = optfn(cost, params, lr, *extra, **kwargs)
    loggedfn.register(cost, params, lr, gn, pn)
    return u, gn, pn

def registered(optfn):
    return lambda *args, **kwargs: register_and_call(optfn, *args, **kwargs)

def get_opt_fn(name):
    if name == 'sgd':
        return registered(sgd)
    elif name == 'rmsprop':
        return registered(rmsprop)
    elif name == 'adagrad':
        return registered(adagrad)
    elif name == 'adam':
        return registered(adam)
    else:
        raise RuntimeError('unrecognized optimizer: %s' % name)
