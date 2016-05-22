import theano
import time
import random
import os
import os.path
import copy_reg

from tables import *

class trainParticle(IsDescription):
    iteration    = Int64Col()
    timestamp    = Float64Col()
    cost         = Float32Col()
    learningRate = Float32Col()
    gradNorm     = Float32Col()
    paramNorm    = Float32Col()

class testParticle(IsDescription):
    iteration    = Int64Col()
    timestamp    = Float64Col()
    cost         = Float32Col()


class FnObj(object):
    def createDbFile(self, fileName):
        if os.path.isfile(fileName):
            self.dbfile = open_file(fileName, "a", title="Job parameters")
            trainTable = self.dbfile.root.parameters.training
            testTable = self.dbfile.root.parameters.validation
        else:
            self.dbfile = open_file(fileName, "w", title="Job parameters")
            dgroup = self.dbfile.create_group("/", 'parameters', 'Dynamic parameters that change with time')
            sgroup = self.dbfile.create_group("/", 'configs', 'Configurations')

            trainTable = self.dbfile.create_table(dgroup, 'training', trainParticle, 'Parameters during training')
            testTable = self.dbfile.create_table(dgroup, 'validation', testParticle, 'Parameters during validation')

        self.trainRow = trainTable.row
        self.testRow = testTable.row

    def __init__(self, logdir):
        self.iters = 0
        self.createDbFile(logdir + "/paramlog.h5")

    def logTest(self, cost):
        self.iters += 1
        self.testRow['iteration'] = self.iters
        self.testRow['timestamp'] = time.time()
        self.testRow['cost'] = cost
        self.testRow.append()

        self.dbfile.flush()

    def logTrain(self, cost, lr, gradNorm, paramNorm):
        self.iters += 1
        self.trainRow['iteration'] = self.iters
        self.trainRow['timestamp'] = time.time()
        self.trainRow['cost'] = cost
        self.trainRow['learningRate'] = lr
        self.trainRow['gradNorm'] =  gradNorm
        self.trainRow['paramNorm'] = paramNorm
        self.trainRow.append()

        self.dbfile.flush()

fnobjs = {}
registered_fns = {}

def register(cost, params, lr, gradNorm, paramNorm):
    registered_fns[cost]=(cost, params, lr, gradNorm, paramNorm)

def log_and_call_train(fnobj, theanofn, *rargs):
    outs = theanofn(*rargs)
    origouts = tuple(list(outs)[4:])
    cost = outs[0]
    lr = outs[1]
    gradNorm = outs[2]
    paramNorm = outs[3]

    if fnobj:
        fnobj.logTrain(cost, lr, gradNorm, paramNorm)

    if len(origouts) == 1:
        return origouts[0]
    if len(origouts) == 0:
        return    
    return origouts

def log_and_call_test(fnobj, theanofn, *rargs):
    outs = theanofn(*rargs)
    origouts = tuple(list(outs)[1:])
    cost = outs[0]

    if fnobj:
        fnobj.logTest(cost)

    if len(origouts) == 1:
        return origouts[0]
    if len(origouts) == 0:
        return    
    return origouts


class CallTest(object):
    def __init__(self, fnobj, theanofn):
        self.fnobj = fnobj
        self.theanofn = theanofn

    def __call__(self, *rargs):
        return log_and_call_test(self.fnobj, self.theanofn, *rargs)

class CallTrain(object):
    def __init__(self, fnobj, theanofn):
        self.fnobj = fnobj
        self.theanofn = theanofn

    def __call__(self, *rargs):
        return log_and_call_train(self.fnobj, self.theanofn, *rargs)

def pickle_CallTest(c):
    return CallTest, (c.theanofn,)

def pickle_CallTrain(c):
    return CallTrain, (c.theanofn,)

copy_reg.pickle(CallTest, pickle_CallTest)
copy_reg.pickle(CallTrain, pickle_CallTrain)

def function(inputs, outputs, mode=None, updates=None, givens=None, no_default_updates=False, accept_inplace=False, name=None, rebuild_strict=True, allow_input_downcast=None, profile=None, on_unused_input='raise'):
    reg = None
    cost = None
    newoutputs = []
    if outputs != None:
        if type(outputs) == list:
            for out in outputs:
                if out in registered_fns:
                    reg = registered_fns[out]
                    cost = out
                    break
        else:
            if outputs in registered_fns:
                reg = registered_fns[outputs]
                cost = outputs

    logdir = os.getenv('JOBDIR')
    if logdir == None:
        logdir = "."

    if reg == None:
        return theano.function(inputs, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input)

    if updates == None:
        # test
        if name == None:
            name = 'test'
    else:
        if name == None:
            name = 'train'

    if type(outputs) == list:
        newoutputs = outputs
    else:
        newoutputs = [outputs]

    if updates == None:
        newoutputs = [reg[0]] + newoutputs
    else:
        newoutputs = [reg[0], reg[2], reg[3], reg[4]] + newoutputs

    theanofn = theano.function(inputs, newoutputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input)

    if cost in fnobjs:
        fnobj = fnobjs[cost]
    else:
        fnobj = FnObj(logdir)
        fnobjs[cost] = fnobj

    if updates == None:
        retfn = CallTest(fnobj, theanofn)
    else:
        retfn = CallTrain(fnobj, theanofn)

    return retfn
