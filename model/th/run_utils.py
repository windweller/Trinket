import os
import logging
import json
from os.path import join as pjoin
import cPickle as pickle
from ug_utils import load_model_params


def get_logger(expdir):
    # log to file and console
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(
        pjoin(expdir, 'log.txt')
    )
    consoleHandler = logging.StreamHandler()
    logger.addHandler(handler)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger


def setup_exp(args):
    expdir = args.expdir
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    logger = get_logger(expdir)

    # save / load training parameters
    if not args.resume_epoch:
        opts = vars(args)
        with open(pjoin(expdir, 'opts.json'), 'w') as fout:
            json.dump(opts, fout, sort_keys=True, indent=4)
    else:
        with open(pjoin(expdir, 'opts.json'), 'r') as fin:
            opts = json.load(fin)
        for k in opts:
            # NOTE be wary of opts besides epoch/resume_epoch that shouldn't overwrite
            if 'epoch' not in k and 'decay_after' not in k:
                setattr(args, k, opts[k])
        logger.info('loaded+new opts:')
        logger.info(opts)
        with open(pjoin(args.expdir, 'opts.json'), 'w') as fout:
            json.dump(vars(args), fout, sort_keys=True, indent=4)

    with open(pjoin(expdir, 'opts.json'), 'w') as fout:
        json.dump(opts, fout, sort_keys=True, indent=4)

    return logger, opts

# decprecated
def load_model_and_opts(args, logger):
    logger.info('loading model...')
    with open(args.model, 'rb') as fin:
        model = pickle.load(fin)
    logger.info('done loading model')
    with open(pjoin(args.expdir, 'opts.json'), 'r') as fin:
        opts = json.load(fin)
        logger.info(opts)
    return model, opts


# deprecated
def restore_model_and_opts(model_fn, args, logger):
    class OptsStruct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    with open(pjoin(args.expdir, 'opts.json'), 'r') as fin:
        opts = json.load(fin)
        logger.info(opts)
    opts_struct = OptsStruct(**opts)
    model = model_fn(opts_struct)
    logger.info('loading model...')
    load_model_params(model, args.model)
    logger.info('done loading model')
    return model, opts
