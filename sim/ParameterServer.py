from .Agent import Agent
from multiprocessing import Process, Pipe, Lock
#import multiprocessing
#import gc
import numpy as np
# from random import Random
import pickle
import pandas as pd
#import globals as g
#from pathlib import Path
#from os.path import join
import os
import math
#import time
#from multiprocessing.sharedctypes import RawArray
#import ctypes
from .Shard import Shard
import util.Logger as logger
import time
from util.Saver import Saver
class ParameterServer(object):
    """maintains weights of the distrubuted optimizer """

    def __init__(self, flags, update_function, agent_function, data_loader, num_agents, slow_down, algorithm):
        """
        :param flags: Flags set by the user.
        :param update_function: Implementation of the UpdateFunction abstract class.
        :param agent_function: Implementation of the abstract AgentFunction class.
        :param data_loader: Function which returns ((train_data, train_labels),(test_data, test_labels),
                number of distinct labels (Categories)).
                If left None, then the data needs to be loaded by the agent_function.
        :param num_agents: The number of Agents to be simulated.
        :param slow_down: How much the agent gets slowed down, either a tuple (float a, float b),
                a single float b or None for no slowdown.
        :param algorithm:  Implementation of the AlgorithmFunction abstract class.
        """
        # flag used to figure out if the PS finished without an interruption
        self.orderly = False
        self.flags = flags
        self.batch_size = self.flags.batch_size // num_agents
        self.num_agents = num_agents
        self.slow_down = slow_down
        self.iterations = self.flags.total_iterations
        logger.state("Number of iterations per agent " + str(self.iterations))
        self.total_updates = self.iterations*self.num_agents
        self.algorithm = algorithm
        self.shards = self.flags.shards
        self.number_of_labels = self.flags.number_of_labels

        #if Dataset should get distributed
        self.distribute = self.flags.distribute

        self.update_rule_uninitiated = update_function
        self.agent_function = agent_function
        self.data_loader = data_loader

        # list that stores gradients of the agents
        self.gradients_list = []
        # pipes to communicate with the agent processes
        self.pipes = []
        # Each agent gets an process same order as self.agents
        self.threads = []

        # list of the weights
        self.weights_numpy = []

        # Calculate what fraction of the gpu memory each Agnet and Shard gets to use
        if flags.gpu_number == 1:
            self.fraction = (1 - 0.01) / (self.num_agents + self.shards)
        if flags.gpu_number > 1:
            half = math.ceil((self.num_agents + self.shards) / flags.gpu_number)
            self.fraction = (1 - 0.01) / (half)

        if self.distribute:
            ((ts, tl), (es, el), number_of_labels) = self.data_loader()

            self.data = ts
            self.labels = tl
            self.test_data = es
            self.test_labels = el
            self.number_of_labels = number_of_labels
            #Distributing the data set such that each Agent gets the same number of distinct samples
            self.data_list, self.labels_list = self._distribute_data()
        else:
            self.data_list = [None for i in range(0, self.num_agents)]
            self.labels_list = [None for i in range(0, self.num_agents)]
            self.test_data=None
            self.test_labels=None

        # getting the initial weighs
        # We use a seperate process here because loading tensorflow in this process sometimes causes problems later on.
        (ps_end, agent_end) = Pipe()
        if self.distribute:
            p = Process(target=create_weights, args= (self.flags, self.agent_function, self.test_data, self.test_labels
                                                 , self.flags.batch_size,agent_end))
        else:
            p = Process(target=create_weights, args=(self.flags, self.agent_function, None,None
                                                          , self.flags.batch_size, agent_end))
        p.daemon = True
        p.start()
        agent_end.close()
        self.weights_numpy = ps_end.recv()
        p.join()
        ps_end.close()

        # list that stores the portion of the weight each shard gets
        self.weights_numpy_list = []
        # number of trainable weights each shard gets, the remainder gets distributed to the first few shards
        size = len(self.weights_numpy) // self.shards
        remainder = len(self.weights_numpy) % self.shards
        offset = 0
        for i in range(0, self.shards):
            if i < remainder:
                new_offset = offset + size + 1
            else:
                new_offset = offset + size
            self.weights_numpy_list.append(self.weights_numpy[offset:new_offset])
            offset = new_offset

        self.pipes_list = []
        # Creating the agent processes
        for i in range(0, self.num_agents):
            pipes_for_agent = []
            pipes_for_shars = []
            for j in range(0, self.shards):
                (ps_end, agent_end) = Pipe()
                pipes_for_agent.append(agent_end)
                pipes_for_shars.append(ps_end)
            self.pipes_list.append(pipes_for_shars)
            p = Process(target=Agent, args=(self.flags,  self.num_agents, i + 1, self.data_list[i],
                                            self.labels_list[i], self.number_of_labels, self.batch_size,
                                            self.weights_numpy_list, self.total_updates, pipes_for_agent,
                                            self.agent_function,
                                            self.fraction, self.slow_down[i], os.getpid()))
            #p.daemon = True, cannot set this because Pytorch uses processes fo its own.
            p.start()
            agent_end.close()
            self.threads.append(p)

        self.shards_list = []
        self.shard_pipes = []

        # running the parameter server shards
        for i in range(0,self.shards):
            (ps_end, shard_end) = Pipe()
            self.shard_pipes.append(ps_end)
            pipes = []
            for j in range(0,self.num_agents):
                pipes.append(self.pipes_list[j][i])
            p = Process(target=Shard, args=(self.flags, pipes,self.weights_numpy_list[i] ,i, self.algorithm,
                                            self.update_rule_uninitiated, self.flags.train_set_size, shard_end,
                                            self.fraction, os.getpid()))

            p.start()
            shard_end.close()
            self.shards_list.append(p)

        for i in self.pipes_list:
           for p in i:
              p.close()

        # delete the reference to the data such that it doesn't use up memory, is not used here anymore
        self.data_list = []
        self.labels_list = []
        self.data = None

        self.csv_name = os.path.sep + self.flags.log_file_names + ".log"
        saver = Saver(self.flags.log_dir + self.csv_name)


        # Receiving the weights, other global variables and metrics from each shard

        # list that stores the weights at the end of each epoch for the test set evaluation
        self.test_weights = []
        # other globals variables needed for the test set evaluation
        self.test_globals = []
        print("getting the test weights", flush=True)
        batchsize = self.flags.evaluation_batchsize
        self.evaluation_steps = self.flags.test_set_size // batchsize
        self.eval_obj = self.agent_function(self.flags, self.test_data, self.test_labels, batchsize, 0, 1, True)
        log = []
        self.weights = []
        self.globals = []
        self.metrics = []
        itr = True
        if self.flags.load_save:
            self.counter = self.flags.starting_epoch
        else:
            self.counter = 0
        while itr:
            self.counter += 1
            self.test_globals = [[]] * self.shards
            self.test_weights = [[]] * self.shards
            c = 0
            while c<self.shards:
                for i,p in enumerate(self.shard_pipes):
                    if p.poll():
                        logger.debug("polled true",flush=True)
                        tmp = p.recv()
                        nr = i
                        c += 1
                        print("nr", nr, flush=True)
                        if tmp == 'end':
                            p.close()
                            del(self.shard_pipes[i])
                            self.shut_down_shards()
                            itr = False
                            break
                        weights_list, moving_avg_list = tmp
                        self.test_globals[nr] = moving_avg_list
                        self.test_weights[nr] = weights_list
                        break
                else:
                    #logger.debug("not polled any", flush=True)
                    time.sleep(0.05)
                if itr ==False:
                    break
            if itr:
                for p in self.shard_pipes:
                    p.send("beginning")
                self.test_weights_unified = self.test_weights[0]
                for i in range(1, self.shards):
                    self.test_weights_unified += self.test_weights[i]
                self.test_averages_unified = self.test_globals[0]  # could also take the mean of the list
                if self.flags.eval_at_end:
                    self.weights.append(self.test_weights_unified)
                    self.globals.append(self.test_averages_unified)

                else:
                    res = self.test_evaluation()

                # stroing the weights, globals, and metrics just in case we need them at a later point again
                if self.flags.save_weights:
                    if self.counter > 5:
                        os.remove(self.flags.saves_dir + os.path.sep + self.flags.dump_file_name + '_' + str(
                            self.counter - 5) + ".pkl")
                    f = open(self.flags.saves_dir + os.path.sep + self.flags.dump_file_name + '_' + str(
                        self.counter) + ".pkl", 'wb')
                    pickle.dump([self.test_weights, self.test_globals], f)

                for p in self.shard_pipes:
                    p.send("finished")
                for p in self.shard_pipes:
                    train = p.recv()
                if self.flags.eval_at_end:
                    self.metrics.append(train)
                else:
                    loglist = [self.counter]
                    for t in train:
                        loglist.append(t)
                    res = res.tolist()
                    for r in res:
                        loglist.append(r)
                    log.append(loglist)
                    saver.save_1D(loglist)
                    logger.results(*loglist)
        if self.flags.eval_at_end:
            for epoch, weight in enumerate(self.weights):
                self.test_weights_unified = weight
                self.test_averages_unified = self.globals[epoch]
                res = self.test_evaluation()
                loglist = [epoch]
                for t in self.metrics[epoch]:
                    loglist.append(t)
                res = res.tolist()
                for r in res:
                    loglist.append(r)
                log.append(loglist)
                saver.save_1D(loglist)
                logger.results(*loglist)

        for t in self.shards_list:
            t.join()
        for t in self.threads:
            t.join()

        self.eval_obj.close()
        saver.close()
        logger.debug("at the end", flush=True)
        self.orderly = True

    def shut_down_shards(self):
        for i,p in enumerate(self.shard_pipes):
            tmp = p.recv()
            if tmp == 'end':
                p.close()
                del (self.shard_pipes[i])


    def _distribute_data(self):
        """
        Splits up the trainset data and labels into chuncks of self.chunk_size.
        Ignores the remainder if it cannot be distributed equally
        :return: data_list: list of chunks of the train data
                 labels_list: list of chunks of the train labels
        """
        data_list = []
        labels_list = []
        chunk_size = self.flags.train_set_size // self.num_agents
        for i in range(0, self.num_agents):
            data_chunk = self.data[i * chunk_size: chunk_size * (i + 1), :]
            data_list.append(data_chunk)
            label_chunk = self.labels[i * chunk_size: chunk_size * (i + 1)]
            labels_list.append(label_chunk)

        return data_list, labels_list

    def test_evaluation(self):
        """
            Does the evaluation of the test set
        """
        logger.debug("starting evaluation")

        weights = self.test_weights_unified
        other_vars = self.test_averages_unified
        list1 = []
        for i in range(0, self.evaluation_steps):
            res = self.eval_obj.evaluate(weights, other_vars)
            list1.append(res)
        tmp1 = np.mean(list1, axis=0)

        return tmp1

    def __del__(self):
        logger.debug("del PS")
        if not self.orderly:
            for p in self.shards_list:
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        raise Exception
            for p in self.threads:
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        raise Exception


def create_weights(flags, agent_function, test_data, test_labels, batch_size,pipe):
    """
    Initializes the weights of the model
    :param agent_function: Implementation of the abstract AgentFunction class
    :param test_data: data of the test set
    :param test_labels: Labels ot the test set
    :param batch_size: batch size to be used
    :param pipe: pipe to send back the data
    """
    if flags.load_save:
        logger.state("Loading saved weights at:", flags.saved_weights)
        f = f = open(flags.saved_weights, "rb")
        ret = pickle.load(f)
        weights = ret[0][0]
    else:
        logger.debug("Before calling the agent_function")
        af = agent_function(flags, test_data, test_labels, batch_size, 0,1, True, None, 1)
        logger.debug("After calling the agent_function")
        weights = af.get_weight()
        af.close()
    if flags.eamsgd:
        weights.append(np.array([-0.05],np.float32)) #inital learning rate for eamsgd

    pipe.send(weights)
    pipe.close()
