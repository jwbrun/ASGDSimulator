import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import matplotlib.gridspec as gridspec

fontsize_titles=14
fontsize_labels = 12
filenames = [
"Softsync_mcas_{1}_times_8_scale=8_#agents={0}:itr=0:spcl=False_mom:agentNr={2}.log",
"Softsync_mcasS_{1}_times_8_scale=8_#agents={0}:itr=0:spcl=False_mom:agentNr={2}.log"]
titles = ["Asynchronous MC", "Asynchronous MC Slow"]
num_agents = 8
Agents = [8]
repeats = 5
def process_data(num_agents, repeats, filename):
    """
        Merges all the TimeStamp files by listing for every agent how many updates it completed
        at every timepoint(Interval)
    :param num_agents:
    :param repeats:
    :param filename: Formatted file name, {0} for the number of Agents, {1} number of repeats, {2} agent nr.
    :return:
    """

    list_of_lists = []
    list_of_means = []
    list_of_stds = []
    first = True
    for k in range(1, num_agents + 1):
        first = True
        for i in range(0, repeats):
            frame = pd.read_csv(filename.format(num_agents, i+1, k))
            if first:
                l = [frame.get_values()[:, 1]]
                print(len(list_of_lists), len(l))
                first= False
            else:
                l.append( frame.get_values()[:, 1])
        list_of_lists.append(l)
    list = []
    s =0
    for l in list_of_lists:
        for ll in l:
            listl = []
            i_old = 0
            for i in ll:
                listl.append(i - i_old)
                i_old = i
            listl = listl[1:]#the first one is a time not an inervall
            list = list + listl
    print(np.mean(list), np.std(list))
    s = len(list)

    mi = np.min(list)
    ma = np.max(list)
    bars = 100
    intervall = (ma - mi) / bars
    print(mi, ma, intervall)

    val_list = [0] * bars
    for i in list:
        ind = math.floor((i - mi) / intervall)
        if ind == bars:
            ind = bars - 1
        val_list[ind] += 1
    val_relative = []
    print("total number of updates", s, flush=True)
    for vl in val_list:
        val_relative.append(vl/s)
    x_dist = [mi + intervall * r + intervall / 2 for r in range(0, bars)]
    print(x_dist, val_list)
    # return x_dist, val_list, intervall
    return x_dist, val_relative, intervall


if __name__ == "__main__":
    plt.figure(figsize=[10, 5])
    plots = [121,122]
    for i, name in enumerate(filenames):
        print(i)
        x_dist, val_list, intervall = process_data(num_agents, repeats, name)
        plt.subplot(plots[i])
        plt.title(titles[i], fontsize=fontsize_titles)
        plt.bar(x_dist, val_list, width=intervall)
        plt.xlabel("Time", fontsize=fontsize_labels)
    plt.tight_layout()
    plt.show()