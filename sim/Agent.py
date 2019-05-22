from multiprocessing import Pipe
from os import getpid
import numpy as np
import time
import pandas as pd
from pathlib import Path
import random
import os
import util.Logger as logger
from util.Saver import Saver
import signal
import math


class Agent(object):
    """
        Implementation of the Agent
    """

    def __init__(self, flags, num_agents, agent_nr, data, labels, num_labels, batch_size, weights, total_updates, pipes,
                 agent_function, fraction, slow_down, ps_pid):
        """
        :param flags: Flags set by the user
        :param num_agents: Number of agents
        :param agent_nr: This agent's number (starting from 1)
        :param data: Test dataset, None if agent_function should load it itself
        :param labels: Labels of the test dataset, None if agent_function should load labels itself
        :param num_labels: Number of distinct labels (categories)
        :param batch_size: Batchsize to be used
        :param weights: Initial weights for the agent_function model
        :param total_updates: Total number of updates all agents do together,
                (Not known apriori how many updates this agent does)
        :param pipes: Pipes that connecting agents to shards (one pipe per shard) len(pipes) == #shards
        :param agent_function: Implementation of the abstract AgentFunction class
        :param fraction: Fraction of the GPU memory allowed to use
        :param slow_down: How much the agent gets slowed down, either a tuple (a,b), a float b or None for no slowdown
        :param ps_pid: Pid of the parameter server
        """
        # flag used to figure out if the agent finished without an interruption
        self.orderly = False
        os.sched_setaffinity(0, list(range(0, flags.threads)))
        logger.state("affinity agent", os.sched_getaffinity(0))
        self._ps_pid = ps_pid
        self.flags = flags
        self.num_agents = num_agents
        self.agent_nr = agent_nr
        self.data = data
        self.labels = labels
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.weights = weights
        logger.state("Agent nr", agent_nr, "weights lenght", len(self.weights))
        self.total_updates = total_updates
        logger.state("Agent nr", agent_nr, "agent batch size", batch_size, flush=True)
        self.slow_down = slow_down
        self.timing = self.flags.time_program  # The time calls them self not put into an if because it almost safes no
        self.random = random.Random()
        seed = (int(time.time() * 100) * agent_nr) % (2 ** 32 - 1)
        self.random.seed(seed)
        self.pipes = pipes

        # Not known apriori how many updates this agent does, so we reserve enoough space.
        # Amount of used memory so far was no problem
        # But incase there is not enough memory this should be changed!
        # +10 because if there is only one agent  then total_updates +1 are done
        self.log = np.zeros((self.total_updates + 10, 2), dtype=np.float64)
        self.num_shards = len(self.pipes)
        self.agent_function = agent_function

        self.fraction = fraction
        self.gpu_nr = (self.agent_nr + self.flags.shards) % self.flags.gpu_number

        self.csv_name = os.path.sep + self.flags.time_stamps_file_name + ":agentNr=" + str(self.agent_nr) + ".log"
        self.saver = Saver(self.flags.log_dir + self.csv_name)
        if not self.flags.load_save:
            self.saver.header(["TimeStamp", "ClockStamp"])

        logger.debug("Agent nr", agent_nr, " with slow down " + str(self.slow_down))
        self.avg = []
        self.finished = []

        self.times = []
        logger.debug("affinity agent: ", self.agent_nr, os.sched_getaffinity(0))
        self.p = self.flags.p
        self.work()
        self.saver.close()
        self.orderly = True

    def work(self):
        """
            Trains for one batch using the self.agent_function and then sends the gradient back to
            the parameter shards and gets an up-to-date weight back
            It logs the time when it finished each update.
            in a file stored at self.flags.time_stamps_file_name
        """
        # store the global update of each weight
        if self.flags.load_save:
            self.batch_size = self.flags.batch_size // self.num_agents
            # number of iteration for the total batch size (Batch size if there were one agent)
            self.iterations = self.flags.total_iterations
            # how many iterations there are per epoch
            if self.flags.drop_remainder:
                self.iterations_in_epoch = math.floor(self.flags.train_set_size / (self.num_agents * self.batch_size))
            else:
                self.iterations_in_epoch = math.ceil(self.flags.train_set_size / (self.num_agents * self.batch_size))
            self.iterations_in_epoch //= 1
            global_update = [self.flags.starting_epoch * self.iterations_in_epoch * self.num_agents] * self.num_shards
        else:
            global_update = [0] * self.num_shards

        #list that sotres the size of each weight in bytes, needed for sending them
        float_sizes = []
        total_size = []
        shapes = []
        weights = []
        for w in self.weights:
            weights += w
            shap = []
            fs = []
            ts = 0
            for i in w:
                l = 1
                shap.append(i.shape)
                for r in i.shape:
                    l *= r
                fs.append(l)
                ts += l
            shapes.append(shap)
            float_sizes.append(fs)
            total_size.append(ts)
        print(total_size[0], flush=True)
        print(float_sizes[0], flush=True)
        print(shapes[0], flush=True)

        logger.debug("Agent nr", self.agent_nr, "pid is: ", getpid(), flush=True)
        if self.data is not None:
            logger.debug("Agent nr", self.agent_nr, self.data.shape, self.labels.shape, self.num_labels,
                         self.batch_size, flush=True)
        agent = self.agent_function(self.flags, self.data, self.labels, self.batch_size, self.gpu_nr, self.fraction,
                                    False, weights, self.agent_nr)
        del (self.data)
        del (self.labels)

        itr = True
        step = 0
        while itr:
            start_time = time.time()
            weights = []
            for w in self.weights:
                weights += w

            grads_, metrics = agent.train(weights)

            agent_part_finished = time.time()
            if self.slow_down != None:
                if type(self.slow_down) is tuple:
                    if self.flags.slow_down_type == "gauss":
                        r = self.random.gauss(self.slow_down[0], self.slow_down[1])
                        if r <= 0:
                            time.sleep(0)
                        else:
                            time.sleep(r)
                else:
                    if self.flags.slow_down_type == "ber":
                        r = self.random.uniform(0, 1)
                        if self.p >= r:
                            print(r)
                            time.sleep(self.slow_down)
                    elif self.flags.slow_down_type == "time":
                        time.sleep(self.slow_down)

            offset = 0
            for c, q in enumerate(self.pipes):
                new_offset = offset + len(self.weights[c])
                for bits in grads_[offset:new_offset]:
                    q.send_bytes(bits)
                q.send((global_update[c], step, self.agent_nr, metrics))
                offset = new_offset
                logger.debug("agent", self.agent_nr, "sent to", c)

            itr_inner = 0
            while itr_inner < self.num_shards:
                for c, p in enumerate(self.pipes):
                    if p.poll():
                        while True:
                            logger.debug("Agent", self.agent_nr, "is recv")
                            tmp = p.recv()
                            logger.debug("Agent", self.agent_nr, "has recv")
                            if tmp[0] == "update weights":
                                global_update[c] = p.recv()
                                weight = []
                                for fsi, fs in enumerate(float_sizes[c]):
                                    w = p.recv_bytes(fs * 4)
                                    weight.append(np.ndarray(shapes[c][fsi], np.float32, w))
                                self.weights[c] = weight
                                itr_inner += 1
                                break
                            elif tmp[0] == "stop":
                                logger.debug("Agent nr", self.agent_nr, "got stoped", flush=True)
                                self.pipes[c].close()
                                del (self.pipes[c])
                                self.shut_down()
                                itr = False
                                itr_inner = self.num_shards
                                break
                            elif tmp[0] == "globals":
                                logger.debug("agent is sending globals")
                                glob = agent.get_globals()
                                p.send(glob)
                else:
                    time.sleep(0.0001)

            self.log[step, 0] = time.time()
            self.log[step, 1] = time.clock()
            self.saver.save_1D(self.log[step, :])
            step += 1
            end_time = time.time()
            if self.timing:
                self.times.append([agent_part_finished - start_time, end_time - start_time])
        logger.debug("Agent nr", self.agent_nr, "loop actually finished", flush=True)

        agent.close()
        for p in self.pipes:
            p.close()
        if self.timing:
            t = np.average(self.times, axis=0)
            logger.state("Agent nr", self.agent_nr, "running agent function:", t[0], "total time", t[1], flush=True)

    def shut_down(self):
        for c, p in enumerate(self.pipes):
            tmp = p.recv()
            if tmp[0] == "stop":
                self.pipes[c].close()
                del (self.pipes[c])

    def __del__(self):
        logger.debug("del Agent")
        if not self.orderly:
            os.system('pkill -TERM -P ' + str(self._ps_pid))
            pass
