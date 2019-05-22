import time
from random import Random
import numpy as np
import util.Logger as logger
from AlgorithmFunction import AlgorithmFunction

class Synchronous(AlgorithmFunction):
    def __init__(self, pipes, float_sizes, shapes, flags, agents):
        super().__init__(pipes, float_sizes, shapes)
        self.softsync = Softsync(pipes, float_sizes, shapes, flags, 1, agents, False, 0)

    def __call__(self, epoch, update):
        return self.softsync.__call__(epoch, update)



class Softsync(AlgorithmFunction):
    """
    Based on the n-softsync algorithm:
    https://arxiv.org/pdf/1511.05950.pdf
    """
    def __init__(self, pipes, float_sizes, shapes,  flags, n, learners, warm_start=False, epochs = 0):
        """
            Waits for learners // n gradients before it returns them to the shard.
         :param pipes: One pipe for each agent
         :param float_sizes: Array which stores for each weight its size in bytes
         :param shapes: Array which stores for each weight its shape
         :param flags: flags set by the user
         :param n: n of n-softsync.
         :param learners: number of agents
         :param warm_start: if the first epochs synchronous should be used
         :param epochs: number of epochs for which synchronous is used
         """
        super().__init__(pipes, float_sizes, shapes)
        assert (learners % n  == 0) #otherwise this algorithm won't finish!

        self.learners = learners
        self.num = learners // n
        self.flags = flags
        self.timing = self.flags.time_program
        self.qlist = []
        self.counter = 0
        self.Random = Random(time.time())
        self.shuffle = flags.shuffle
        self.warm_start = warm_start
        self.epochs = epochs

        self.steps_for_learner = [0] * learners
        self.max_staleness = 0
        self.gen = self._pipes_generator()

    def _pipes_generator(self):
        while True:
            if self.shuffle:
                self.Random.shuffle(self.pipes)
            for i, p in enumerate(self.pipes):
                yield (i,p)


    def __call__(self, epoch, update):
        """
            Implementation with the learners waiting n/learners others to arrive
        """
        count = 0
        ii = 1

        gradients_list = []
        metrics_list = []
        from_list = []
        step_list = []
        global_update_list = []

        while True:
            i,p = next(self.gen)
            if p.poll():
                count += 1
                grads  =[]
                for i,fs in enumerate(self.float_sizes):
                    w = p.recv_bytes(fs*4)
                    grads.append(np.ndarray(self.shapes[i],np.float32, w))

                last_update, step, agnt_nr, metrics = p.recv() #only marginal gains her in the e-05s not worth the complexity to doing it with recv_bytes

                gradients_list.append(grads)
                metrics_list.append(metrics)
                from_list.append(agnt_nr)
                global_update_list.append(last_update)
                step_list.append(1)
            else:
                ii += 1
            if ii %self.learners == 0:
                time.sleep(0.0001)

            if self.warm_start and self.epochs >= epoch:
                if count == self.learners:
                    return gradients_list, from_list, global_update_list ,step_list,  metrics_list, 0, 2
            else:
                if count == self.num:
                    return gradients_list, from_list, global_update_list,step_list, metrics_list, 0, 2

    def __del__(self):
        pass

class softsync_holdback(AlgorithmFunction):
    """
    If one agent is ahead of others by more than a set maximum, then it is not allowed to do further updates until the
    others catch up.
    Implementation based on SSP:
    https://papers.nips.cc/paper/4894-more-effective-distributed-ml-via-a-stale-synchronous-parallel-parameter-server.pdf
    """
    def __init__(self, pipes, float_sizes, shapes, flags, n, learners, max_difference=1):
        """
            :param pipes: One pipe for each agent
            :param float_sizes: Array which stores for each weight its size in bytes
            :param shapes: Array which stores for each weight its shape
            :param warm_start: if the first epochs synchronous should be used
            :param epochs: number of epochs for which synchronous is used
            :param flags: flags set by the user
            :param n: n of n-softsync.
            :param learners: number of agents
            :param max_difference: maximal difference the fastes agent can be ahead
            """
        super().__init__(pipes, float_sizes, shapes)


        assert (learners % n  == 0) #otherwise this algorithm won't finish!!!
        self.learners = learners
        self.num = learners // n
        self.flags = flags
        self.timing = self.flags.time_program
        self.qlist = []
        self.counter = 0
        logger.state("softsync with number of learners", learners)
        self.Random = Random(time.time())
        self.shuffle = flags.shuffle
        self.updates_per_learner = [0]*learners
        self.min = 0
        self.max_difference = max_difference
        self.mode = 0
        self.holdback = []
        self.gen = self._pipes_generator()

    def _pipes_generator(self):
        while True:
            if self.shuffle:
                self.Random.shuffle(self.pipes)
            for i, p in enumerate(self.pipes):
                yield (i, p)


    def __call__(self, epoch, update):
        count = 0
        ii = 1
        gradients_list = []
        metrics_list = []
        from_list = []
        step_list = []
        global_update_list = []


        if self.mode == 1:
            agnt_nr = self.holdback.pop()
            logger.state("poped", agnt_nr, flush=True)
            if len(self.holdback) == 0:
                self.mode = 0
            return  None, [agnt_nr], None, None, None, 0, 1

        while True:
            i, p = next(self.gen)
            if p.poll():
                grads = []
                for i, fs in enumerate(self.float_sizes):
                    w = p.recv_bytes(fs * 4)
                    grads.append(np.ndarray(self.shapes[i], np.float32, w))

                last_update, step, agnt_nr, metrics = p.recv()

                sending = 2
                if self.mode == 0:
                    if self.updates_per_learner[agnt_nr-1] >= self.min + self.max_difference:
                        self.holdback.append(agnt_nr)
                        sending = 0
                        logger.debug("holding back", agnt_nr, flush=True)
                    else:
                        self.updates_per_learner[agnt_nr - 1] += 1
                        new_min = np.min(self.updates_per_learner)
                        if new_min != self.min:
                            self.min = new_min
                            if len(self.holdback) != 0:
                                self.mode = 1
                        logger.debug("reg send", agnt_nr, flush=True)
                    count += 1
                    gradients_list.append(grads)
                    metrics_list.append(metrics)
                    from_list.append(agnt_nr)
                    global_update_list.append(last_update)
                    step_list.append(1)
                else:
                    ii += 1
                if ii % self.learners == 0:
                    time.sleep(0.0001)

                if count == self.num:
                    if self.timing:
                        self.counter += 1
                    return gradients_list, from_list, global_update_list,step_list, metrics_list, 0, sending

    def __del__(self):
        pass



class softsync_binned(AlgorithmFunction):
    """
    The gradients get put into different bins according to their staleness. Each bin has its own weights which get
    only updated by gradients put in there. Bins get averaged after a certain number of updates.
    """
    def __init__(self, pipes, float_sizes, shapes, flags, n, learners, bins):
        """
            :param pipes: One pipe for each agent
            :param float_sizes: Array which stores for each weight its size in bytes
            :param shapes: Array which stores for each weight its shape
            :param warm_start: if the first epochs synchronous should be used
            :param epochs: number of epochs for which synchronous is used
            :param flags: flags set by the user
            :param n: n of n-softsync.
            :param learners: number of agents
            :param bins: Array representing the boundaries along which the gradients get binned.
        """

        super().__init__(pipes, float_sizes, shapes)

        assert (learners % n  == 0) #otherwise this algorithm won't finish!!!

        self.learners = learners
        self.num = 1
        self.flags = flags
        self.timing = self.flags.time_program
        self.qlist = []
        self.counter = 0
        self.Random = Random()
        self.shuffle = flags.shuffle
        self.bins = bins
        self.bin_counts = [0]*(len(bins)+1)

        self.gen = self._pipes_generator()

        # Only works for one update at the time

    def _pipes_generator(self):
        while True:
            if self.shuffle:
                self.Random.shuffle(self.pipes)
            for i, p in enumerate(self.pipes):
                yield (i, p)


    def __call__(self, epoch, update):
        """
            Implementation with the learners waiting n/learners others to arrive
        """
        ii=1
        count = 0
        list = []
        gradients_list = []
        metrics_list = []
        from_list = []
        step_list = []
        global_update_list = []
        while True:
            i, p = next(self.gen)
            if p.poll():
                grads = []
                for i, fs in enumerate(self.float_sizes):
                    w = p.recv_bytes(fs * 4)
                    grads.append(np.ndarray(self.shapes[i], np.float32, w))

                last_update, step, agnt_nr, metrics = p.recv()

                count += 1

                gradients_list.append(grads)
                metrics_list.append(metrics)
                from_list.append(agnt_nr)
                global_update_list.append(last_update)
                step_list.append(1)
                staleness = update - last_update
            else:
                ii += 1
            if ii % self.learners == 0:
                time.sleep(0.0001)
            if count == self.num:
                binning = 0
                for i in self.bins:
                    if staleness >= i:
                        binning += 1
                    else:
                        break
                self.bin_counts[binning] += 1
                logger.debug("staleness", staleness, "put in bin", binning, flush=True)
                return gradients_list, from_list, global_update_list, step_list, metrics_list, binning, 2

    def __del__(self):
        if self.timing:
            logger.state("bin counts", self.bin_counts)


class softsync_accrued(AlgorithmFunction):
    """
    A phase is the time it takes for every agent to submit an update. If an agents submits more than one update in a
    phase then the second and all consequent gradients are hold back until the end of the phase.
    """
    def __init__(self, pipes, float_sizes, shapes, flags, n, learners):
        """
            :param pipes: One pipe for each agent
            :param float_sizes: Array which stores for each weight its size in bytes
            :param shapes: Array which stores for each weight its shape
            :param warm_start: if the first epochs synchronous should be used
            :param epochs: number of epochs for which synchronous is used
            :param flags: flags set by the user
            :param n: n of n-softsync.
            :param learners: number of agents
        """
        super().__init__(pipes, float_sizes, shapes)

        assert (learners % n  == 0) #otherwise this algorithm won't finish!!!
        self.learners = learners
        self.num = learners // n
        self.flags = flags
        self.timing = self.flags.time_program
        self.qlist = []
        self.counter = 0
        logger.state("softsync with number of learners", learners)
        self.Random = Random(time.time())
        self.shuffle = flags.shuffle
        self.step_in_phase = [0] * learners
        self.steps_for_learner = [0] * learners
        self.min = 0
        self.mode = 2
        self.holdback = []
        self.max_staleness = 0
        self.gen = self._pipes_generator()

    def _pipes_generator(self):
        while True:
            if self.shuffle:
                self.Random.shuffle(self.pipes)
            for i, p in enumerate(self.pipes):
                yield (i, p)


    def __call__(self, epoch, update):
        count = 0
        ii=1
        gradients_list = []
        metrics_list = []
        from_list = []
        steps_list = []
        global_update_list = []

        if self.mode == 0:
            bb =self.holdback[0]
            del(self.holdback[0])
            for i in range(0,1):#len(bb)):
                tmp = bb#[i]  # want a queue and not a stack such that we process in arriving order
                #del (self.holdback[i])
                last_update, counter,  step, agnt_nr, grads,  metrics = tmp
                logger.debug("poped", agnt_nr, step, flush=True)

                gradients_list.append(grads)
                metrics_list.append(metrics)
                from_list.append(agnt_nr)
                global_update_list.append(last_update)
                steps_list.append(step)

            if len(self.holdback) == 0:
                self.mode = 2
                self.step_in_phase = [0] * self.learners
                self.holdback = []
            return  gradients_list, from_list,global_update_list, steps_list,  metrics_list, 0, 0  # now what we want when self.num != 1
        while True:
            i, p = next(self.gen)
            if p.poll():
                count += 1
                grads = []
                for i, fs in enumerate(self.float_sizes):
                    w = p.recv_bytes(fs * 4)
                    grads.append(np.ndarray(self.shapes[i], np.float32, w))

                last_update, step, agnt_nr, metrics = p.recv()

                self.step_in_phase[agnt_nr - 1] += 1
                self.steps_for_learner[agnt_nr-1] += 1
                new_min = np.min(self.step_in_phase)
                self.max_staleness = np.max(self.steps_for_learner)

                if self.step_in_phase[agnt_nr - 1] > self.min+1: #big problem is once one agent is ahead by a few steps then it will always have its gradients accrued until it is falling behing again
                    self.holdback.append([last_update, self.steps_for_learner[agnt_nr -1 ], self.step_in_phase[agnt_nr - 1], agnt_nr, grads, metrics])
                    sending = 1
                    logger.debug("accruing", agnt_nr, self.step_in_phase[agnt_nr - 1], flush=True)
                else:
                    sending = 2
                    #not needed if if case above is true, here anyway
                    logger.debug("doing normal", agnt_nr, self.step_in_phase[agnt_nr - 1],  flush=True)

                gradients_list.append(grads)
                metrics_list.append(metrics)
                from_list.append(agnt_nr)
                global_update_list.append(last_update)
                steps_list.append(self.step_in_phase[agnt_nr - 1])

                if new_min != self.min:
                    self.min = new_min
                    if len(self.holdback) != 0:  # only here such that we don't unecessarly change the mode
                        self.mode = 0
                        j = np.ceil(len(self.holdback) / self.learners)
                        #self.holdback = [self.holdback[i * self.learners:i * self.learners + self.learners] for i in range(0, int(j))]
                    else:
                        self.step_in_phase = [0] * self.learners
                        self.holdback = []

            else:
                ii += 1
            if ii % self.learners == 0:
                time.sleep(0.0001)
            if count == self.num:
                if self.timing:
                    self.counter += 1
                    if self.counter %1000==0:
                        logger.state("qsize:", np.mean(self.qlist[-100*self.num:]), flush=True)
                return gradients_list, from_list, global_update_list, steps_list, metrics_list, 0, sending

    def __del__(self):
        pass
