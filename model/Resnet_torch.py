#for the parts from https://github.com/eladhoffer/convNet.pytorch
#commit: 298fb46b6229b06a399efbb544dbcbf9f5d98032
"""
MIT License

Copyright (c) 2017 Elad Hoffer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from AgentFunction import AgentFunction
from convNet.models.resnet import ResNet_cifar
from convNet.utils.cross_entropy import CrossEntropyLoss
from convNet.data import DataRegime
import os
import torch
import numpy as np

import time
import math
from torch.nn.utils import clip_grad_norm_
from convNet.utils.meters import accuracy #check if this does not average
from convNet.utils.param_filter import FilterParameters, is_not_bn, is_not_bias
import torch.nn as nn
from matplotlib import pyplot as plt
import util.Logger as logger


#do we make all the updates
class AgentResnetFunction(AgentFunction):
    def __init__(self, flags, data, labels, batch_size, gpu_nr, fraction, testing, weights=None, agent_nr=1):
        """
        :param flags: flags set by the user
        :param data: sampels if None have to be loaded in the by the actual implementation
        :param labels: lables  if None have to be loaded in the by the actual implementation
        :param batch_size: batch size to be used
        :param gpu_nr: Number of the GPU, the agent should run on
        :param fraction: fraction of the GPU memory the agent should maximally use
        :param testing: True if we run a test set, False for training
        """
        super().__init__(flags, data, labels, batch_size, gpu_nr, fraction, testing, weights, agent_nr)
        seed = (int(time.time()*100)*agent_nr) % (2**32 - 1)
        self.timess = [] 
        logger.debug("seed", seed)
        np.random.seed(seed)
        #random.seed(123)
        torch.manual_seed(seed)  #set randomness for everything that torch uses the pythono random for. Here padding, image order
        self.flags = flags
        self.resnet = ResNet_cifar(depth=44)#num_classes = 100
        #self.resnet = ResNet_imagenet(block=BasicBlock,
         #                      layers=[2, 2, 2, 2],
          #                     expansion=1)
        self.loss = CrossEntropyLoss()
        self.batch_size = batch_size
        self.gpu_nr = gpu_nr
        logger.debug("number of gpus", torch.cuda.device_count())

        if 'cuda' in self.flags.device:
            self.device = torch.device(self.gpu_nr)
            logger.debug("using device ", torch.cuda.get_device_name(self.device),"am Agent ",agent_nr)
        else:
            self.device = torch.device('cpu')

        self.cpu = torch.device('cpu')

        self.resnet.to(self.device, dtype = torch.float)
        self.loss.to(self.device)
        path = os.environ['HOME']
        path = path + '/cifar10_data/'
        logger.debug("gpu_nr", gpu_nr)

        # copied and modified from
        # convNet.pytorch/utils/regularization   Regularizer class
        # and convNet.pytorch/resnet: weight_decay_config function
        self._named_parameters = list(
            FilterParameters(self.resnet, **{'parameter_name': lambda n: not n.endswith('bias'),
                                             'module': lambda m: not isinstance(m, nn.BatchNorm2d)}).named_parameters())

        # copid and modified from:
        # convNet.pytorch/main.py
        if not testing:
            self.train_set =  DataRegime(getattr(self.resnet, 'data_regime', None),
                       defaults={'datasets_path':path , 'name': 'cifar10', 'split': 'train',
                                 'augment': True,
                                 'input_size': None, 'batch_size': batch_size, 'shuffle': True,
                                 'num_workers': 1, 'pin_memory': True, 'drop_last': True,
                                 'distributed': False, 'duplicates': 1, #batch augmentation
                                 'cutout': {'holes': 1, 'length': 16} if True else None})
            self.train_epoch = 0
            self.train_iterator = self._train_generator()
        else:
            self.test_set = DataRegime(getattr(self.resnet, 'data_eval_regime', None),
                                  defaults={'datasets_path': path, 'name': 'cifar10', 'split': 'val',
                                            'augment': False,
                                            'input_size': None, 'batch_size': batch_size,
                                            'shuffle': False,
                                            'num_workers': 1, 'pin_memory': True, 'drop_last': False})
            self.test_epoch = 0
            self.test_iterator = self._test_generator()
        #end


        if flags.load_save:
            import pickle
            f = f = open(flags.saved_weights, "rb")
            ret= pickle.load(f)
            glob = ret[1][0]
            for i, p in enumerate(self.resnet.buffers()):
                if np.isscalar(glob[i]):
                    p.data = torch.tensor(glob[i])
                else:
                    p.data[:] = torch.tensor(glob[i], device=self.device)[:]




        if self.flags.eamsgd and not testing:
            weights = weights[0:-1]
        if (self.flags.correction or self.flags.eamsgd) and not testing:
            if weights != None:
                self.velocity = []
                self.velocity_grad = []
                self.momentum = torch.tensor(0.9, dtype=torch.float, device = self.device)
                for w in weights:
                    self.velocity.append(torch.zeros(w.shape,dtype=torch.float, device = self.device))


    def _train_generator(self):
        # adopted from this trainer implementation
        # convNet.pytorch/trainer.py
        while True:
            self.train_set.set_epoch(self.train_epoch)
            self.train_set_load = self.train_set.get_loader().__iter__() #load every epoch a new one
            for inputs_batch, target_batch in self.train_set_load:
                inputs_flat = inputs_batch
                target_flat = target_batch
                for inputs, target in zip(inputs_flat.chunk(1, dim=0),
                                          target_flat.chunk(1, dim=0)):
                    yield (inputs,target)
            self.train_epoch += 1

    def train(self, weights):
        """
        Do one more forward propagation and calculate the gardient
        :param weights: The weights to be loaded before beginning
        :return:    grads: calculated gradient, as numpy representation
                    globals: other global variables, as numpy representation
                    metrics: list of metrics
        """
        if self.flags.eamsgd:
            lr = torch.tensor(weights[-1],dtype=torch.float, device = self.device)
            weights = weights[0:-1]
        t1 = time.time()
        for i,p in enumerate(self.resnet.parameters()):
            p.data.copy_(torch.from_numpy(weights[i]), non_blocking=True)
        t2 = time.time()
        self.timess.append(t2-t1)


        #based on this trainer implementation
        #adopted from this trainer implementation
        #convNet.pytorch/trainer.py,
        self.resnet.zero_grad()
        outputs = []
        total_loss = 0
        self.resnet.train() #from Trainer
        inputs, target = self.train_iterator.__next__()

        target = target.to(self.device)
        inputs = inputs.to(self.device, dtype = torch.float)

        output = self.resnet(inputs)
        loss = self.loss(output, target)
        grad = None

        if isinstance(output, list) or isinstance(output, tuple):
            output = output[0]

        outputs.append(output.detach())

        loss.backward()

        total_loss += float(loss)


        outputs = torch.cat(outputs, dim=0)
        prec1, prec5 = accuracy(outputs, target, topk=(1, 5))
        # end of code from the trainer
        grads = []
        #copied and modified from convNet.pytorch/utils/L2Regularization
        #L2Regularization calss, pre_step
        for n,p in self._named_parameters:
            d_p = p.grad.data
            d_p.add_(1e-4, p.data)
        # print(cc, flush=True)
        for i,p in enumerate(self.resnet.parameters()):
            d_p = p.grad.data
            if self.flags.correction:
                #momentum correction
                #based on https://arxiv.org/pdf/1712.01887.pdf

                self.velocity[i].mul_(self.momentum).add_(d_p)
                tmp = self.velocity[i].mul(self.momentum).add_(d_p)
                grads.append(tmp.cpu().numpy())
            elif self.flags.eamsgd:
                # eamsgd
                # based on https://arxiv.org/pdf/1412.6651.pdf
                dplr =d_p.mul(lr)
                self.velocity[i].mul_(self.momentum).add_(dplr)
                tmp = self.velocity[i].mul(self.momentum).add_(dplr)
                p.data.add_(tmp)
                grads.append(p.data.cpu().numpy())
            else:
                grads.append(d_p.cpu().numpy())



        return grads, [total_loss,float(prec1)]


    def get_globals(self):
        globals = []
        for i, p in enumerate(self.resnet.buffers()):
            globals.append(p.cpu().numpy())
        return globals

    def get_weight(self):
        """
        :return: the current weight of the agent
        """
        list = []
        counter = 0
        for p in self.resnet.parameters():
            #print(p)
            #p.cpu()
            list.append(p.cpu().detach().numpy())
            counter +=1

        logger.debug("counter:", counter)
        return list

    def _test_generator(self):
        # based on this trainer implementation
        # adopted for the purpose here
        # convNet.pytorch/master/trainer.py
        while True:
            self.test_set.set_epoch(self.test_epoch)
            self.test_set_load = self.test_set.get_loader().__iter__()  # load every epoch a new one
            for inputs_batch, target_batch in self.test_set_load:
                inputs_flat = inputs_batch
                target_flat = target_batch
                for inputs, target in zip(inputs_flat.chunk(1, dim=0),
                                          target_flat.chunk(1, dim=0)):
                    yield (inputs, target)

            self.test_epoch += 1


    def evaluate(self, weights, globals):
        """
        :param weights: weights to be loaded
        :param globals: global variables to be loaded
        :return: list of metrics
        """
        if self.flags.eamsgd:
            weights = weights[0:-1]
        for i,p in enumerate(self.resnet.parameters()):
            p.data.copy_(torch.from_numpy(weights[i]), non_blocking=True)
        for i, p in enumerate(self.resnet.buffers()):
            if np.isscalar(globals[i]):
                p.data= torch.tensor(globals[i])
            else:
                p.data.copy_(torch.from_numpy(globals[i]), non_blocking=True)
        self.resnet.to(self.device)

        #based on this trainer implementation
        #adopted for the purpose here
        #convNet.pytorch/trainer.py
        outputs = []
        total_loss = 0
        self.resnet.eval()
        inputs, target = self.test_iterator.__next__()
        target = target.to(self.device)
        inputs = inputs.to(self.device, dtype=torch.float)

        output = self.resnet(inputs)
        loss = self.loss(output, target)

        if isinstance(output, list) or isinstance(output, tuple):
            output = output[0]

        outputs.append(output.detach())

        total_loss += float(loss)

        outputs = torch.cat(outputs, dim=0)
        prec1, prec5 = accuracy(outputs, target, topk=(1, 5))
        # end

        return [total_loss, float(prec1)]

    def close(self):
        """
        need to close the data loaders otherwise they keep running
        """
        logger.debug("close resnet torch", flush=True)
        if self.timess != []:
            logger.state("time is: ",np.mean(self.timess, axis = 0), flush=True )
        a = getattr(self, "test_set_load", None)
        b = getattr(self,"train_set_load", None)
        if a != None:
            logger.debug("shut them down",flush=True)
            # del(a)
            a._shutdown_workers()
        if b != None:
            logger.debug("shut them down")
            b._shutdown_workers()
