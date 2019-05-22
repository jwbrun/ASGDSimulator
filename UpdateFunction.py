import abc


class UpdadteFunction(abc.ABC):

    @abc.abstractmethod
    def __init__(self, flags, agents, gpu_nr, memory_fraction, weights):
        """
        :param flags: Flags set by the user
        :param agents: Number of agents
        :param gpu_nr: GPU to be used
        :param memory_fraction: Fraction of the GPU that can be used
        :param weights: Copy of the model weights
        """
        super().__init__()
        self.flags = flags
        self.agents = agents
        self.gpu_nr = gpu_nr
        self.memory_fraction = memory_fraction
        self.weight = weights

    @abc.abstractmethod
    def __call__(self, weights, update, gradients, staleness, epoch):
        """
        :param weights: Copy of the model weights
        :param update: Curent update
        :param gradients: List of gradients
        :param staleness: Staleness of each gradient
        :param epoch: Current epoch
        """
    pass

    @abc.abstractmethod
    def close(self):
        """
        Closes all open resources
        """
    pass
