import abc


class AlgorithmFunction(abc.ABC):
    def __init__(self, pipes, float_sizes, shapes):
        """
        :param pipes: One pipe for each agent
        :param float_sizes: Array which stores for each weight its size in bytes
        :param shapes: Array which stores for each weight its shape
        """
        super().__init__()
        self.pipes = pipes
        self.float_sizes = float_sizes
        self.shapes = shapes

    @abc.abstractmethod
    def __call__(self, epoch, update):
        """
        :param epoch: Current epoch
        :param update: Curent update
        :return:
            gradients_list: List containing the gradients
            from_list: To which agent each gradient belongs
            global_update_list: Update number tag of every gradient
            step_list: List with additional information that gets logged
            metrics_list: List containing a list of metrics
            binning: Bin number to update
            sending: 0: Shard only does the update, but does not send back its weights to the agents, 1: only send back weights, 2: both
        """
        pass