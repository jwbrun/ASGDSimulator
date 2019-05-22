from multiprocessing import Process, Pipe, Lock, Queue
import numpy as np
import os
import math
import time
import util.Logger as logger
from util.Saver import Saver


class Shard(object):
    def __init__(self, flags, pipes, weights, shard_nr, algorithm, update_function, dataset_size, pipe, fraction,
                 ps_pid):
        """
        Simulates a shard
        :param flags: Flags set by the user
        :param pipes: Pipes connecting agents and shards (one pipe per agent) len(pipes) == #agents
        :param weights: Initial weights for this shard
        :param shard_nr: Shard number (starting form 0)
        :param algorithm:  Implementation of the abstract AlgorithmFunction class.
        :param update_function:  Implementation of the abstract AgentFunction class
        :param dataset_size: Number of samples in the data-set
        :param pipe: Pipe to communicate with the ParameterServer
        :param fraction: Fraction of the GPU allowed to use
        :param ps_pid: PID of the ParameterServer, used to shut down the program in case of an error.
        """

        # flag used to figure out if the shard finished without an interruption
        self.orderly = False
        os.sched_setaffinity(0, list(range(0, flags.threads)))
        logger.state("affinity shard", os.sched_getaffinity(0))
        self._ps_pid = ps_pid
        self.flags = flags
        self.pipes = pipes
        self.shard_nr = shard_nr
        self.num_agents = len(self.pipes)  # constraint to be imposed
        self.update_function_uninitiated = update_function
        self.times = []
        self.dataset_size = dataset_size
        self.pipe = pipe
        self.fraction = fraction

        self.timing = self.flags.time_program  # The time calls them self not put into an if because it almost safes no
        # time if timing is false, and costs if timing True
        self.printing = self.flags.printing
        self.print_interval = self.flags.print_interval

        #period after which bins get averaged
        self.bins_period = self.flags.bins_period

        #Variables used to calculate the staleness
        self.steps_for_learner = [0] * self.num_agents
        self.max_staleness = 0

        # Logs the training metrics, used to send them back at the end to the PS
        self.training_metrics_log = []
        # Logs the weights, used to send them back at the end to the PS
        self.evaluation_weights = []

        self.bins = self.flags.bins
        # Putting the weights into shared memory such that the update_function and Communicator can use the same copy
        self.weights = []
        for i in range(0, self.bins):
            self.weights.append(self.create_copy(weights))


        self.float_sizes = []
        self.total_size = 0
        self.shapes = []
        for i in self.weights[0]:
            self.shapes.append(i.shape)
            l = 1
            for r in i.shape:
                l *= r
            self.float_sizes.append(l)
            self.total_size += l

        self.algorithm = algorithm(self.pipes, self.float_sizes, self.shapes)

        self.batch_size = self.flags.batch_size // self.num_agents
        # number of iteration for the total batch size (Batch size if there were one agent)
        self.iterations = self.flags.total_iterations
        # number of iteration we do in total (taking number of agents into account)
        self.total_updates = self.iterations * self.num_agents
        # how many iterations there are per epoch
        if self.flags.drop_remainder:
            self.iterations_in_epoch = math.floor(self.dataset_size / (self.num_agents * self.batch_size))
        else:
            self.iterations_in_epoch = math.ceil(self.dataset_size / (self.num_agents * self.batch_size))
        logger.state("Shard {0} is doing epochs:{1}".format(shard_nr, self.iterations // self.iterations_in_epoch))
        # here for testing
        self.iterations_in_epoch //= 1  # if changed here also has to be changed in the agent

        self.csv_name_staleness = os.path.sep + self.flags.staleness_file_name + "_shard" + str(self.shard_nr) + ".log"
        self.saver = Saver(self.flags.log_dir + self.csv_name_staleness)
        if not self.flags.load_save:
            self.saver.header(["Agent", "staleness", "phase_staleness", "global_staleness"])
        logger.state("Shard {0} set up saving".format(shard_nr))

        logger.state("Shard {0} set up the communicator".format(shard_nr))
        logger.debug("affinity shard", os.sched_getaffinity(0))
        self._run_shard()
        self.shut_down()
        self.pipe.send("end")
        if self.bins > 1:
            logger.state("the bins were", self.algorithm.bin_counts)

        # self.queue.close()
        for p in self.pipes:
            p.close()
            logger.debug(p.closed)

        self.pipe.close()
        logger.debug(self.pipe.closed)
        self.orderly = True

    def __del__(self):
        logger.debug("del Shard", self.shard_nr)
        if not self.orderly:
            os.system('pkill -TERM -P ' + str(self._ps_pid))

    def create_copy(self, weights):
        shared_list = []
        for i in weights:
            array = np.copy(i)
            shared_list.append(array)

        return shared_list

    def _run_shard(self):

        # staleness log, stores the staleness of each update
        staleness_log = np.empty((self.iterations * self.num_agents, 4), np.int64)
        update_rule = []
        for i in range(0, self.bins):
            update_rule.append(
                self.update_function_uninitiated(self.flags, self.num_agents, (self.shard_nr) % self.flags.gpu_number,
                                                 self.fraction, self.weights[i]))
        if self.flags.load_save:
            grad_update = self.flags.starting_epoch * self.iterations_in_epoch * self.num_agents
            epoch = self.flags.starting_epoch + 1  # Start at one, for the update function
            validation_checker = 0
        else:
            grad_update = 0
            epoch = 1  # Start at one, for the update function
            # second counter that gets reset once we finished an epoch
            validation_checker = grad_update
        # used to do stuff during the processing of the first update
        first = True
        # how many updates were done for each bin
        bins_update = [0] * self.bins

        # flag set if we need to wait for PS
        wait_ps = False

        # how many metrics have been added up
        metrics_count = 0

        # number of updates in an epoch
        updates_until_validation = self.iterations_in_epoch * self.num_agents
        print_abs = updates_until_validation // self.print_interval
        if print_abs ==0:
            print_abs = 1
        start_time = time.time()
        while True:
            logger.debug("shard", self.shard_nr, " does next iter")
            before_weights_time = time.time()
            #calling the algorithm to get the gradients
            gradients_list, from_list, tag_list, step_list, metrics_list,\
                        binning, sending = self.algorithm(epoch, grad_update)
            logger.debug("shard", self.shard_nr, " after algorithm")
            if sending != 1:
                got_weights_time = time.time()
                if first:
                    # list for all the moving average variant of the metrics
                    moving_averages = [0] * len(metrics_list[0])
                    first = False

                logger.debug("shard", self.shard_nr, "before loop")

                staleness_list = []
                for j, anr in enumerate(from_list):
                    self.steps_for_learner[anr - 1] += 1
                    staleness_list.append(grad_update - tag_list[j]+1)
                self.max_staleness = np.max(self.steps_for_learner)
                send_back = [agnr -1 for agnr in from_list]
                updates  = len(from_list)

                for j, s in enumerate(from_list):
                    staleness_log[grad_update + j, 3] = self.max_staleness - self.steps_for_learner[s - 1]
                    staleness_log[grad_update + j, 2] = step_list[j]
                    staleness_log[grad_update + j, 1] = staleness_list[j] -  1
                    staleness_log[grad_update + j, 0] = from_list[j]  # agent nr
                    self.saver.save_1D(staleness_log[grad_update + j, :])
                logger.debug("shard", self.shard_nr, " after loop")
                # Perform the update of the weights, note the weights and update are in shared memory and get updated in-suto
                ww = update_rule[binning](self.weights[binning], grad_update, gradients_list, staleness_list, epoch)
                if not self.flags.eamsgd:
                    self.weights[binning] = ww
                logger.debug("shard", self.shard_nr, " after update rule")
                bins_update[binning] += 1

                # updates stores how many gradients get used in the same update
                grad_update += updates
                validation_checker += updates

                # calculating moving average of the metrics
                for metrics in metrics_list:
                    for mi, mv in enumerate(metrics):
                        moving_averages[mi] += mv
                metrics_count += updates
                #printing the current average train metrics
                if self.printing and self.shard_nr == 0 and validation_checker % print_abs == 0:
                    logger.results(str(grad_update) + "/" + str(updates_until_validation),
                                   *np.divide(moving_averages, metrics_count),
                                   staleness_list[-1])

                if self.flags.bins >1 and grad_update % self.bins_period == 0:
                    self.elastic(bins_update)
                    bins_update = [0] * self.bins

                if validation_checker >= updates_until_validation:
                    weights_copy = self.weights[binning]

                    self.evaluation_weights.append(weights_copy)
                    self.training_metrics_log.append(np.divide(moving_averages, metrics_count))

                    validation_checker -= updates_until_validation
                    logger.results(epoch, *np.divide(moving_averages, metrics_count))
                    for metrics in metrics_list:
                        for mi, mv in enumerate(metrics):
                            moving_averages[mi] = 0
                    metrics_count = 0
                    logger.state("Time/updates:", (time.time() - start_time) / grad_update, flush=True)
                    epoch += 1

                    i = send_back[0]
                    p = self.pipes[i]
                    logger.debug("shard", self.shard_nr, "before sending to Agent 'globals' to", i)
                    p.send(["globals", weights_copy])
                    logger.debug("shard", self.shard_nr, "after send")
                    glob = p.recv()
                    logger.debug("shard", self.shard_nr, "after recv")
                    self.pipe.send([weights_copy, glob])
                    logger.debug("shard", self.shard_nr, "sent to ps")
                    wait_ps = True

            before_comm = time.time()
            if sending != 0:
                send_back = [agnr - 1 for agnr in from_list]
                if self.flags.eamsgd:
                    logger.state("eamsgd shard", flush=True)
                    weights_copy = ww
                else:
                    weights_copy = self.weights[binning]

                self.send_updates(send_back, "update weights", grad_update, weights_copy)

            if sending != 1:
                end_time = time.time()
                if self.timing:
                    self.times.append(
                        [end_time - before_weights_time, end_time - got_weights_time, end_time - before_comm])

            if wait_ps:
                while True:
                    a = self.pipe.recv()
                    if a == "beginning":
                        logger.debug("received beginning")
                    if a == "finished":
                        logger.debug("received finished")
                        self.pipe.send(self.training_metrics_log[-1])
                        logger.debug("sent training metrics")
                        wait_ps = False
                        break

            if not (grad_update < self.total_updates):
                break

        for i in range(0, self.bins):
            update_rule[i].close()
        if self.timing:
            t = np.average(self.times, axis=0)
            logger.state("Shard, total time:", t[0], "without waiting for gradients:", t[1], "comm time: ", t[2])

    def send_updates(self, list, message, grad_update, weights):
        """
            send updates to members of list
        :param list: of the index of the pipes we want to send the updates to.
        :param updates: list of values to send
        """
        for i in list:

            self.pipes[i].send([message])
            self.pipes[i].send(grad_update)
            for fs in weights:
                self.pipes[i].send_bytes(fs)

    def elastic(self, bins_updates):
        # based on https://arxiv.org/pdf/1412.6651.pdf
        logger.debug("elastic", bins_updates, flush=True)
        weights = self.flags.bin_weights
        bins_updates = [u * v for (u, v) in zip(bins_updates, weights)]
        sum = np.sum(bins_updates)
        bins_updates = [i / sum for i in bins_updates]
        logger.debug("elastic", bins_updates, flush=True)

        new_weight = [np.multiply(self.weights[i], m) for i, m in enumerate(bins_updates)]
        new_weight = np.sum(new_weight, axis=0)
        for i, w in enumerate(new_weight):
            for wb in self.weights:
                wb[i][:] = w[:]

    def shut_down(self):
        while len(self.pipes) > 0:
            for i, p in enumerate(self.pipes):
                if p.poll():
                    grads = []
                    for j, fs in enumerate(self.float_sizes):
                        w = p.recv_bytes(fs * 4)
                        grads.append(np.ndarray(self.shapes[j], np.float32, w))
                    p.recv()
                    p.send(["stop"])
                    p.close()
                    del (self.pipes[i])
                    break  # else get out of range error
