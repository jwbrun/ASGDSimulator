import abc


class AgentFunction(abc.ABC):
    @abc.abstractmethod
    def __init__(self, flags, data, labels, batch_size, gpu_nr, memory_fraction, testing, weights, agent_nr):
        """
        :param flags: Flags set by the user
        :param data: Sampels if None have to be loaded
        :param labels: Lables if None have to be loaded.
        :param batch_size: Batch size to be used
        :param gpu_nr: Number of the GPU to use
        :param fraction: Fraction of the GPU memory the agent should maximally use
        :param testing: True if we run a test set, False for training
        :param weights: Copy of the model weights
        :param agent: Number of the agent that initializes this object
        """
        super().__init__()
        self.flags = flags
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.gpu_nr = gpu_nr
        self.memory_fraction = memory_fraction
        self.testing = testing
        self.weights = weights
        self.agent_nr = agent_nr

    @abc.abstractmethod
    def train(self, weights):
        """
        :param weights: Weights to be loaded
        :return gradients: Same shape as the weighs
                metrics: list of train metrics
        """

    @abc.abstractmethod
    def get_weight(self):
        """
        returns a copy of the current weight
        """

    @abc.abstractmethod
    def get_globals(self):
        """
        returns the globals variables of the model
        """
        pass

    @abc.abstractmethod
    def evaluate(self, weights, globals):
        """
        :param weights: Weights to be loaded
        :param globals: Global variables to be loaded
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        Close all open resources
        """
        pass
