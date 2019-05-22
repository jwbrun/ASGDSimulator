import time
from sim.ParameterServer import ParameterServer
import gc
import Optimizers as op
import Algorithms
import argparse
import os
from pathlib import Path
import util.Logger as logger


def main(flags, data_loader, agent_work_function, update_rule, algorithm, ags, i):
    SLOW_AGENTS = [(0,2) for i in range(1, 51)]  # Agents are numbered from 1

    start_time = time.time()
    PS = ParameterServer(flags, update_rule, agent_work_function, data_loader, ags, SLOW_AGENTS, algorithm)
    gc.collect()
    end_time = time.time()
    print("time", end_time - start_time, i)


if __name__ == "__main__":
    handler = argparse.ArgumentParser()
    handler.add_argument('-log_dir', type=str, default=os.environ['HOME'] + '/Logs',
                         help="Location where to log file are stored.")

    handler.add_argument('-saves_dir', type=str, default=os.environ['HOME'] + '/Logs' + '/Saves',
                         help="Location where test weights get stored.")

    handler.add_argument('-log_file_names', type=str, default='Log_softsync',
                         help="Prefix of the Log file that stores the metrics for each epoch.")

    handler.add_argument('-time_stamps_file_name', type=str, default='TimeStamps_softsync',
                         help="Prefix for the TimeStamp files that each agent creates")

    handler.add_argument('-staleness_file_name', type=str, default='staleness_softsync',
                         help="Prefix for the Staleness file")

    handler.add_argument('-dump_file_name', type=str, default="Dump_softsync",
                         help="Prefix for the pickled file that stores all the weights, and train metrics")

    handler.add_argument('-agents', type=int, default=4,
                         help="Number of agents the program simulates")

    handler.add_argument('-runs', type=int, default=1,
                         help="How many times the program should rerun the simulation with the same parameters")

    # one epoch is TRAIN_SET_SIZE / BATCH_SIZE rounded down if the remainder is droped (in the AgentFunction implementation)
    handler.add_argument('-total_iterations', type=int, default=50,
                         help="For how many iterations the simulator runs(if there is more than one agent, this"
                              "number gets multiplied by the number of agents).")
    # should be dividable by the number of agents, else it does less than the intended number of epochs!!
    handler.add_argument('-batch_size', type=int, default=40,
                         help="Total batch-size(if there is more than one agent, this number gets divided by the number"
                              "of agents).")

    handler.add_argument('-gpu_number', type=int, default=2,
                         help="Number of available GPUs.")

    handler.add_argument('-distribute', type=bool, default=False,
                         help="If True the Train set gets loaded and equally distributed to all "
                              "agents(The remainder gets droped).")

    handler.add_argument('-shards', type=int, default=1,
                         help="Number of server shards that get simulated")

    handler.add_argument('-train_set_size', type=int, default=50000,  # 1'281'167 for imagenet
                         help="Number of samples in the train set")
    handler.add_argument('-test_set_size', type=int, default=10000,  # 50000 for imagenet
                         help="Number of samples in the test set.")
    handler.add_argument('-number_of_labels', type=int, default=10,  # 1000
                         help="Number of Cathegories(destinct labels) in the Dataset")

    handler.add_argument('-hogw', type=bool, default=True,
                         help="If True the agent has to wait until the parameter server has processed its gradient "
                              "until it receives new weights")
    handler.add_argument('-evaluation_batchsize', type=int, default=200,
                         help="Batch-size used during evaluating the test set")
    handler.add_argument('-save_weights', type=bool, default=True,
                         help="If True the accrued weights for testing get stored as dump_file_name at saves_dir")

    handler.add_argument('-time_program', type=bool, default=True,
                         help="If True times different parts of the program")
    handler.add_argument('-printing', type=bool, default=True,
                         help="If True prints out information that may help debuging")
    handler.add_argument('-device', type=str, default='cpu',
                         help="Either cpu or cuda")
    handler.add_argument('-drop_remainder', type=bool, default=True,
                         help="Drop the remainder of the batch")
    handler.add_argument('-scale', type=int, default=1)
    handler.add_argument('-shuffle', type=bool, default=False,
                         help="Shuffle the queue")
    handler.add_argument('-correction', type=bool, default=False,
                         help="wheather to apply nestrov momentum correction")
    handler.add_argument('-eamsgd', type=bool, default=False)
    handler.add_argument('-p', type=float, default=0.5,  help="probability that the agent gets slowed down"
                                                              "only used in combination with slow_down_type = Bernulli")

    # flags for saving and restoring weights
    handler.add_argument('-saved_weights', type=str, default=os.environ['HOME'] + '/Saves/save.pkl',
                         help="Location where the saved weights are stored.")
    handler.add_argument('-load_save', type=bool, default=False)
    handler.add_argument('-starting_epoch', type=int, default=1,
                         help="If you have a weight dump for epoch x then this flag should be set to x")
    handler.add_argument('-eval_at_end', type=bool, default=True,
                         help="If True then the simulator does the evaluation on the test set at the end")
    handler.add_argument('-threads', type=int, default=128,
                         help="Some systems set the affinity after every fork back to 1, so set the number of threads" +
                              " to be used here")
    handler.add_argument("-print_interval", type=int, default=1000)
    handler.add_argument('-slow_down_type', type=str, default="gauss",
                         help="ber: Bernulli, gauss: gaussian, time: specified time")
    handler.add_argument('-bins_period', type=int, default=16,
                         help="How many updates pass before the bins get averaged")
    handler.add_argument('-bin_weights', type=float, nargs='*', default=[1.5,1,0.5],
                         help="Number of agents the program simulates")
    handler.add_argument('-bins', type=int, default=3,
                         help="how many bins the shard shall use. Note: to actually make use of them have to use and "
                              "Algorithm that uses them")

    handler.parse_args()
    flags = handler.parse_args()

    try:
        local = Path(flags.log_dir)
        local.mkdir(exist_ok=True)
    except:
        print("Path given to log_dir argument is not a path")
        quit()
    try:
        local = Path(flags.saves_dir)
        local.mkdir(exist_ok=True)
    except:
        print("Path given to saves_dir argument is not a path")
        quit()

    if flags.test_set_size % flags.evaluation_batchsize != 0:
        print("Test set size should be divisible by the evaluation batchsize")
        quit()
    if flags.batch_size % flags.agents != 0:
        print("Batch size should be divisible by the number of agents")
        # otherwise id doesn't do the specified number of iterations
    logger.Logger_setup(gap=20, state=True, debug=False)
    print("begin")
    log_file_name = flags.log_file_names
    print(flags, flush=True)
    ags = flags.agents
    i = 0
    n = ags
    scale = ags
    staleness = True

    os.sched_setaffinity(0, list(range(0, flags.threads)))
    flags.time_stamps_file_name = log_file_name + "_times_" + str(n) + '_' + 'scale=' + str(
        scale) + '_' + "#agents=" + str(ags) + ":itr=" + str(i) + ":spcl=" + str(staleness)
    flags.staleness_file_name = log_file_name + "_stl_" + str(n) + '_' + 'scale=' + str(
        scale) + '_' + "#agents=" + str(ags) + ":itr=" + str(i) + ":spcl=" + str(staleness)
    flags.dump_file_name = log_file_name + "_dump_" + str(n) + '_' + 'scale=' + str(
        scale) + '_' + "#agents=" + str(ags) + ":itr=" + str(i) + ":spcl=" + str(staleness)
    flags.log_file_names = log_file_name + "_log_" + str(n) + '_' + 'scale=' + str(
        scale) + '_' + "#agents=" + str(ags) + ":itr=" + str(i) + ":spcl=" + str(staleness)

    #moment = lambda flag, agents, gpu_nr, memory_fraction, weights: op.MomentumWrapperExp(1,1,12,flags = flag, agents= agents, gpu_nr = gpu_nr,
     #                                                                                         memory_fraction = memory_fraction, weights = weights, staleness_aware=staleness,
      #                                                                                        divider=ags, scale=1)
    moment = lambda flag, agents, gpu_nr, memory_fraction, weights: op.MomentumWrapperCorrection(flags=flag,
                                                                                          agents=agents, gpu_nr=gpu_nr,
                                                                                          memory_fraction = memory_fraction, weights = weights, staleness_aware=staleness,
                                                                                          divider=ags, scale=1)
    sync = lambda a, b, c: Algorithms.Softsync(a, b, c, flags, n, ags, False, 0)  #[2,4])

    from model.Resnet_torch import AgentResnetFunction as AgentFunction


    from model.mnist_deep import AgentMNISTFunction
    #main(flags, None, AgentMNISTFunction, moment, sync, ags, i)
    main(flags, None, AgentFunction, moment, sync, ags, i)
