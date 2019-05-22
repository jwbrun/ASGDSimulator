#for the code snippeds from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
"""
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
import torch
import torch.cuda
import time
from UpdateFunction import UpdadteFunction
import numpy as np
import math
from cifar10fast.core import PiecewiseLinear
import util.Logger as logger

class BaseWrapper(UpdadteFunction):
        def __init__(self, flags=None, agents = 1, gpu_nr = 0, memory_fraction=1, weights= None, divider=1,
                     staleness_aware=False, epochs=0, scale=1, nesterov=True, momentum = 0.9,
                     lrSchedule=[(1, 0.1), (6, 0.1), (31, 0.01), (61, 0.001), (81, 0.0001)]):
            """

            :param flags: Flags set by the user
            :param agents: Number of agents
            :param gpu_nr: GPU to be used
            :param memory_fraction: Fraction of the GPU that can be used
            :param weights: Copy of the model weights
            :param divider: Learning rate gets divided by this. Set this to the number of agents to use learning rate scaling.
            :param staleness_aware: If True then the learning rate is scaled by the staleness.
            :param epochs: Number of epochs to to warm-up
            :param scale: Learning rate gets multiplied by this factor
            :param nesterov: If True then nesterov momentum is used
            :param momentum: Momentum value for momentum SGD
            :param lrSchedule: Learning rate schedule, list of (epoch, lr) tuples,  epoch starts from 1!
            """
            super().__init__(flags, agents, gpu_nr, memory_fraction, weights)
            self.times = []
            self.divider = divider
            self.staleness_aware = staleness_aware
            self.nesterov = nesterov
            self.lr_dict = iter(lrSchedule)
            self.next_switch = next(self.lr_dict)
            self.learning_rate = 0
            self.batchsize = self.flags.batch_size
            self.epochs = epochs
            self.divider = scale

            if self.flags.drop_remainder:
                self.iterations_in_epoch = self.agents * math.floor(self.flags.train_set_size / (self.flags.batch_size))
            else:
                self.iterations_in_epoch = self.agents * math.ceil(self.flags.train_set_size / (self.flags.batch_size))

            logger.state("iterations in epoch", self.iterations_in_epoch)
            ##
            if 'cuda' in self.flags.device:
                self.device = torch.device('cuda:' + str(self.gpu_nr))
            else:
                self.device = torch.device('cpu')
            self.buf = []
            for w in weights:
                self.buf.append(np.zeros_like(w))
            self.momentum = momentum
            self.lr = PiecewiseLinear([0, epochs], [0.1, 0.1 * self.divider])

        def learning_rate_func(self, epoch, update):
            if self.next_switch is not None and epoch >= self.next_switch[0]:
                logger.state("change of learning rate", epoch, self.learning_rate, self.next_switch[1])
                self.learning_rate = self.next_switch[1]

                logger.state("lr", self.learning_rate)
                try:
                    self.next_switch = next(self.lr_dict)
                except StopIteration:
                    self.next_switch = None
            if self.epochs >= epoch:
                return self.lr(update / self.iterations_in_epoch) / self.divider
            else:
                return self.learning_rate / self.divider * self.divider

        def __call__(self, weights, update, gradients, staleness, epoch):
            """
            :param weights: Copy of the model weights
            :param update: Curent update
            :param gradients: List of gradients
            :param staleness: Staleness of each gradient
            :param epoch: Current epoch
            """
            pass



class MomentumWrapperExp(BaseWrapper):
    """
        Optimizer that keeps an history of the past staleness and fits a Gaussian curve to it and scales the learning
        rate by it.
    """

    def __init__(self, mean, std, ring_size, **arguments):
        """

        :param mean: Mean Staleness to bootstrap the process
        :param std: STD of the Staleness
        :param ring_size: Size of the ring that stores the staleness history
        :param arguments: arguments for the BaseWrapper see documentation there
        """
        super().__init__(**arguments)
        self.use_exp = self.staleness_aware
        self.ring_size = ring_size
        self.staleness_counter = 0
        self.std = std
        self.mean = mean
        self.staleness_ring = np.zeros(self.ring_size, int)
        self.lr_ring = np.zeros(self.ring_size, np.float32)
        self.learn = 0.1  # 1/self.ring_size
        self.decay = 0.9  # 1 - self.learn
        self.equalizer = 1 / self.ring_size


        self.weights = []
        self.buf = []
        for w in self.weight:
            t = torch.from_numpy(w).to(device = self.device)
            self.weights.append(t)

            self.buf.append( torch.zeros_like(t.data, device = self.device))
            self.momentum = torch.tensor(0.9, device = self.device)
            self.lr = PiecewiseLinear([0, 5, 24], [0, 0.4, 0])
            #self.lr = lambda x: (np.piecewise(x, [x<82,x>=82 and x<123,x>=123 and x<165, x>=165],[0.5,0.05,0.005,0.0005])).item(0)

    def learning_rate_func(self, epoch, update):
        if  self.next_switch is not None and epoch >= self.next_switch[0]:
            logger.state("change of learning rate", epoch, self.learning_rate, self.next_switch[1])
            self.learning_rate = self.next_switch[1]/self.divider
            logger.state("lr", self.learning_rate)
            try:
                self.next_switch = next(self.lr_dict)
            except StopIteration:
                self.next_switch = None

        return self.learning_rate

    def __call__(self, weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """

        self.learning_rate_func(epoch, update)
        lr = torch.tensor(- self.learning_rate, dtype = torch.float, device = self.device)

        start_time = time.time()
        if not self.use_exp:
            gradient = gradients
        else:
            lr_list = []

            def exp(x):
                return self.equalizer * np.exp(-(x - self.mean) ** 2 / (2 * self.std ** 2))

            for i, s in enumerate(staleness):
                # print(s)
                pos = self.staleness_counter % self.ring_size
                self.staleness_ring[pos] = s
                lrexp = exp(s)
                self.lr_ring[pos] = lrexp
                lr_list.append(lrexp)
                self.staleness_counter += 1
                if pos == self.ring_size - 1:  ##is this at the end of the ring
                    std = np.std(self.staleness_ring)
                    self.std = self.decay * self.std + self.learn * std
                    if self.std == 0:  # not numerically correct
                        self.std = 0.2

                    mean = np.mean(self.staleness_ring)
                    self.mean = self.decay * self.mean + self.learn * mean
                    sum = np.sum(self.lr_ring) #want sum to be about (self.ring_size/self.agents)
                    equalizer = self.equalizer * (self.ring_size/self.agents) / sum  # problme that we change the equalizer and avgs while iterating through the stalenesses???????
                    self.equalizer = self.decay * self.equalizer + self.learn * equalizer
                    logger.debug("mean", self.mean, "std", self.std, "stalleness ring",self.staleness_ring, "lr ring", self.lr_ring, "sum", sum, "equalizer", self.equalizer, flush=True)
            if self.staleness_counter <= self.ring_size * 5:
                gradient = gradients
            else:
                gradient = [np.multiply(g, s) for g, s in zip(gradients, lr_list)]
        grad = np.mean(gradient, axis=0)
        grads = []
        for g in grad:
            grads.append(torch.tensor(g, device = self.device))

        # followig code snipped is based on https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
        for i,p in enumerate(self.weights):
            d_p = grads[i].data
            if self.momentum != 0:
                buf = self.buf[i]
                lr_grad = d_p.mul(lr)
                buf.mul_(self.momentum).add_(lr_grad)#corrected the version of lr
                #buf.mul_(self.momentum).add_(d_p)
                if self.nesterov:
                    d_p = buf.mul(self.momentum).add_(d_p.mul(lr))
                    #d_p = d_p.add(self.momentum, buf)
                else:
                    d_p = buf
            #p.data.add_(lr,d_p)
            p.data.add_(d_p)

        end_time = time.time()
        if self.flags.time_program:
            self.times.append(end_time - start_time)

        w = []
        for wg in self.weights:
            w.append(wg.data.cpu().numpy())
        return w

    def __del__(self):
        if self.flags.time_program:
            if self.times != []:
                t = np.mean(self.times, axis=0)
                logger.state("Optimizer took", t,flush=True)

    def close(self):
        pass


class MomentumWrapperWarm(BaseWrapper):
    """
    Momentum SGD done on the GPU
    """
    def __init__(self, **arguments):
        super().__init__(**arguments)

        self.weights = []
        self.buf = []
        for w in self.weight:
            t = torch.from_numpy(w).to(device = self.device)
            self.weights.append(t)

            self.buf.append( torch.zeros_like(t.data, device = self.device))
            self.momentum = torch.tensor(0.9, device = self.device)
            self.lr = PiecewiseLinear([0, 5, 24], [0, 0.4, 0])
            #self.lr = lambda x: (np.piecewise(x, [x<82,x>=82 and x<123,x>=123 and x<165, x>=165],[0.5,0.05,0.005,0.0005])).item(0)


    def __call__(self, weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """

        lrr = self.learning_rate_func(epoch, update)
        #self.learning_rate = self.lr(update.value / self.iterations_in_epoch)#/self.batchsize

        lr = torch.tensor(- lrr*len(gradients), dtype = torch.float, device = self.device)
        #logger.debug(lr, update.value / self.iterations_in_epoch, "epoch", epoch)

        start_time = time.time()
        if not self.staleness_aware:
            gradient = gradients
        else:
                gradient = [np.divide(g, s) for g, s in zip(gradients, staleness)]
        grad = np.mean(gradient, axis=0)
        grads = []
        for g in grad:
            grads.append(torch.tensor(g, device = self.device))

        #lr = torch.tensor(- self.lr(update.value / self.iterations_in_epoch), device = self.device).to(dtype = torch.float)#/self.flags.batch_size
        # print(lr)

        # followig code snipped is  based on https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
        for i,p in enumerate(self.weights):
            d_p = grads[i].data
            if self.momentum != 0:
                buf = self.buf[i]
                #lr_grad = d_p.mul(lr)
                #buf.mul_(self.momentum).add_(lr_grad)#corrected the version of lr
                buf.mul_(self.momentum).add_(d_p)
                if self.nesterov:
                    #d_p = buf.mul(self.momentum).add_(lr_grad) #should use lr_grad here too, but leave it for now
                    d_p = d_p.add(self.momentum, buf)
                else:
                    d_p = buf
            p.data.add_(lr,d_p)
            #p.data.add_(d_p)
            #checked the alternative implementation, it is correct
        #end
        end_time = time.time()
        if self.flags.time_program:
            self.times.append(end_time - start_time)
        w = []
        for wg in self.weights:
             w.append(wg.data.cpu().numpy())
        return w

    def __del__(self):
        if self.flags.time_program:
            if self.times != []:
                t = np.mean(self.times, axis=0)
                logger.state("Optimizer took", t, flush=True)

    def close(self):
        pass

class MomentumWrapperWarmCpu(BaseWrapper):
    """
    momentum SGD done on the cpu
    """

    def __init__(self, **arguments):
        super().__init__(**arguments)


    def __call__(self, weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """

        lrr = -self.learning_rate_func(epoch, update)*len(gradients)
        print(lrr)
        logger.debug("the learning rate is:",lrr)

        start_time = time.time()
        if not self.staleness_aware:
            gradient = gradients
        else:
                gradient = [np.divide(g, s) for g, s in zip(gradients, staleness)]
        grad = np.mean(gradient, axis=0)
        for i,g in enumerate(grad):
            if self.momentum != 0:
                buf = self.buf[i]
                np.add(np.multiply(self.momentum,buf,out=buf),g,out=buf)
                if self.nesterov:
                    np.add(np.multiply(self.momentum,buf),g, out=g)
                else:
                    g = buf
                np.add(self.weight[i], np.multiply(lrr, g), out = self.weight[i])

        end_time = time.time()
        if self.flags.time_program:
            self.times.append(end_time - start_time)
        return self.weight

    def __del__(self):
        if self.flags.time_program:
            if self.times != []:
                t = np.mean(self.times, axis=0)
                logger.state("Optimizer took", t, flush=True)

    def close(self):
        pass

class MomentumWrapperCorrection(BaseWrapper):
    """
    SGD done on the cpu
    """

    def __init__(self, **arguments):
        super().__init__(**arguments)

    def __call__(self, weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """
        lrr = self.learning_rate_func(epoch, update)*len(gradients)
        #print(lrr, flush=True)


        start_time = time.time()
        if not self.staleness_aware:
            gradient = gradients
        else:
                logger.debug("optimizer staleness list", staleness)
                for i, k in enumerate(staleness):
                    if k < self.agents:
                        staleness[i] = self.agents
                gradient = [np.divide(g, s) for g, s in zip(gradients, staleness)]
        grad = np.mean(gradient, axis=0)

        for l, w in enumerate(weights):
            np.subtract(w, lrr * grad[l], out=w)
     # print("release lock opt", flush=True)
        end_time = time.time()
        if self.flags.time_program:
            self.times.append(end_time - start_time)
        return weights

    def __del__(self):
        if self.flags.time_program:
            if self.times != []:
                t = np.mean(self.times, axis=0)
                logger.state("Optimizer took", t, flush=True)

    def close(self):
        pass


class MomentumWrapperEamsgd(UpdadteFunction):
    """
    Based on this paper https://cs.nyu.edu/~zsx/nips2015.pdf
    """
    def __init__(self, flags, agents, gpu_nr, memory_fraction, weights, scale, staleness_aware):
        super().__init__(flags,agents, gpu_nr, memory_fraction, weights)
        self.times = []
        self.flags = flags
        self.agents = agents
        self.gpu_nr = gpu_nr
        self.scale = scale
        self.staleness_aware = staleness_aware
        self.nesterov = True
        self.lr_dict = iter([(1, 0.4), (82, 0.04), (123, 0.004), (165, 0.0001)])
        self.next_switch = next(self.lr_dict)
        self.learning_rate = 0
        self.batchsize = self.flags.batch_size



        if self.flags.drop_remainder:
            self.iterations_in_epoch =self.agents*  math.floor(self.flags.train_set_size/ (self.flags.batch_size))
        else:
            self.iterations_in_epoch = self.agents* math.ceil(self.flags.train_set_size/ (self.flags.batch_size))

        logger.state("iterations in epoch", self.iterations_in_epoch)
        ##
        if 'cuda' in self.flags.device:
            self.device = torch.device('cuda:'+str(self.gpu_nr))
        else:
            self.device = torch.device('cpu')

    def learning_rate_func(self, epoch, update):
        if  self.next_switch is not None and epoch >= self.next_switch[0]:
            logger.state("change of learning rate", epoch, self.learning_rate, self.next_switch[1])
            self.learning_rate = self.next_switch[1]/self.scale

            logger.state("lr", self.learning_rate)
            try:
                self.next_switch = next(self.lr_dict)
            except StopIteration:
                self.next_switch = None
        return self.learning_rate


    def __call__(self, weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """

        lrr = self.learning_rate_func(epoch, update)
        #self.learning_rate = self.lr(update.value / self.iterations_in_epoch)#/self.batchsize

        alpha =  lrr *0.9/(lrr*self.agents)
        #logger.debug(lrr, alpha, update.value / self.iterations_in_epoch, "epoch", epoch, "eamsgd")

        start_time = time.time()
        if not self.staleness_aware:
            gradient = gradients
        else:
                gradient = [np.divide(g, s) for g, s in zip(gradients, staleness)]
        grad = gradient[0]
        grad_list = []
        for l,w in enumerate(weights[0:-1]):
            tmp = np.multiply(np.subtract(grad[l], w), alpha)
            np.add(w, tmp, out=w)
            grad_list.append(np.subtract(grad[l], tmp))
        end_time = time.time()
        if self.flags.time_program:
            self.times.append(end_time - start_time)
        grad_list.append(-lrr)
        return grad_list

    def __del__(self):
        # pass
        if self.flags.time_program:
            if self.times != []:
                t = np.mean(self.times, axis=0)
                logger.state("Optimizer took", t, flush=True)

    def close(self):
        pass


class YellowFin(UpdadteFunction):
    """
    This class is a wrapper for the YellowFin implementation found at: https://github.com/AnonRepository/YellowFin_Pytorch
    and described in this paper: https://arxiv.org/pdf/1706.03471.pdf
    Because the code is not distributed under any licence, we cannot include it in the project.
    to make this project work download the tuner_utils folder from the github page and add it to the project folder.
    Allso has to be patched to work with newer versions of pytorch. (add .numpy to line 275 and 283)
    """


    def __init__(self, flags, agents, gpu_nr, memory_fraction, weights, lock, scale, staleness_aware, epoch):
        super().__init__(flags,agents, gpu_nr, memory_fraction, weights, lock)
        from tuner_utils.yellowfin import YFOptimizer
        self.times = []
        self.flags = flags
        self.agents = agents
        self.gpu_nr = gpu_nr
        self.lock = lock
        self.scale = scale
        self.staleness_aware = staleness_aware
        self.nesterov = True
        self.lr_dict = iter([(1, 0.4), (82, 0.04), (123, 0.004), (165, 0.0001)])
        self.next_switch = next(self.lr_dict)
        self.learning_rate = 0
        self.batchsize = self.flags.batch_size
        self.epochs = epoch


        if self.flags.drop_remainder:
            self.iterations_in_epoch =self.agents*  math.floor(self.flags.train_set_size/ (self.flags.batch_size))
        else:
            self.iterations_in_epoch = self.agents* math.ceil(self.flags.train_set_size/ (self.flags.batch_size))

        logger.state("iterations in epoch", self.iterations_in_epoch)
        if 'cuda' in self.flags.device:
            self.device = torch.device('cuda:'+str(self.gpu_nr))
        else:
            pass
        self.device = torch.device('cpu')

        self.weight = []
        self.buf = []
        for w in weights:
            t = torch.tensor(w, device = self.device, requires_grad=True)
            t.grad = torch.tensor(w, device = self.device)
            self.weight.append(t)

        self.optimizer = YFOptimizer(self.weight, lr = 1, mu=0.9)

        self.optimizer.zero_grad()


    def learning_rate_func(self, epoch, update):
        if  self.next_switch is not None and epoch >= self.next_switch[0]:
            logger.state("change of learning rate", epoch, self.learning_rate, self.next_switch[1])
            self.learning_rate = self.next_switch[1]

            logger.state("lr", self.learning_rate)
            try:
                self.next_switch = next(self.lr_dict)
            except StopIteration:
                self.next_switch = None
        if self.epochs >= epoch:
            # print("warm up optimizer", flush=True)
            return self.learning_rate
        else:
            return self.learning_rate/self.scale


    def __call__(self, weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """

        lrr = self.learning_rate_func(epoch, update)

        lr = torch.tensor(- lrr*len(gradients), dtype = torch.float, device = self.device)
        self.optimizer.set_lr_factor(lrr)


        start_time = time.time()
        if not self.staleness_aware:
            gradient = gradients
        else:
                gradient = [np.divide(g, s) for g, s in zip(gradients, staleness)]
        grad = np.mean(gradient, axis=0)


        i = 0
        for elem in self.optimizer._optimizer.param_groups:
            for p in elem['params']:
                p.grad.data.copy_(torch.from_numpy(grad[i]), non_blocking=True)
                i +=1

        self.optimizer.step()


        c=0
        wei = []
        with self.lock:
            for elem in self.optimizer._optimizer.param_groups:
                for p in elem['params']:

                    wei.append(p.data.numpy())
                    c +=1
        end_time = time.time()
        if self.flags.time_program:
            self.times.append(end_time - start_time)
        return wei

    def __del__(self):
        if self.flags.time_program:
            if self.times != []:
                t = np.mean(self.times, axis=0)
                logger.state("Optimizer took", t,flush=True)

    def close(self):
        pass


from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
class AdamWrapper(UpdadteFunction):
    def __init__(self,  flags, agents, gpu_nr, memory_fraction, weights):
        super().__init__(flags, agents, gpu_nr, memory_fraction, weights)
        self.adam = AdamOptimizer(weights, learning_rate_init=1e-3 / self.agents)  #

    def __call__(self,  weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """

        grads = np.mean(gradients, axis=0)
        self.adam.update_params(grads)
        return self.adam.params

    def close(self):
        pass
