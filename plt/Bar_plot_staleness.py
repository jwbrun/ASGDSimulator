import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

staleness_type = 2 #2 for the old tyoe 3 for the new type
"""
file_names = ['staleness_softsync8_scale=1_#agents=8:itr=0:spcl=False_shard0.log',
'staleness_softsync8_scale=8_#agents=8:itr=0:exp=True_shard0.log',
'staleness_softsync8_scale=8_#agents=8:itr=0:spcl=False_bin_slow_shard0.log',
'staleness_softsync8_scale=8_#agents=8:itr=0:spcl=False_cor_slow_shard0.log',
'staleness_softsync8_scale=8_#agents=8:itr=0:spcl=Falsestl_corr_half_shard0.log',
'staleness_softsync8_scale=8_#agents=8:itr=0:spcl=Falsestl_corr_shard0.log',
'staleness_softsync8_scale=8_#agents=8:itr=0:spcl=Falsestl_shard0.log']
"""
file_names = [
            'Softsync_async_fast_{0}_stl_16_scale=16_#agents=16:itr=0:spcl=False_mom_shard0.log',
            'Softsync_async_{0}_stl_16_scale=16_#agents=16:itr=0:spcl=False_mom_shard0.log',
            'Softsync_async_stepwise_{0}_stl_16_scale=1_#agents=16:itr=0:spcl=True_mom_shard0.log']
#titles= ["Hogwild not scaled","Hogwild Exp","Binning slow","Momentum Correction slow","Staleness Corrected half", "Staleness Corrected","Staleness"]
titles= ["Asynchronous MC","Asynchronous MC Stl", "Asynchronous MC SwStl","Asynchronous MC Bin"]

num_Runs = 1
Agent = []
over_all = True
scales = []
fontsize_titles=14
fontsize_labels = 12

#plt.title("Staleness Distributions", fontsize=16)
plt.figure(figsize=[15,5])
# easiest way to center the last row
# https://stackoverflow.com/questions/52014678/aligning-a-row-of-plots-in-matplotlib
gs = gridspec.GridSpec(1,8)
dims = [gs[0,0:2], gs[0,2:4], gs[0,4:6],gs[0,6:8]]


for i,names in enumerate(file_names):
    agent_nrs = []
    stalenesses = []
    for j in range(0, num_Runs):
        # staleness_softsync4_scale=4_#agents=40:itr=0
        data = pd.read_csv(names.format(j+1))
        agent_nrs.append(data.get_values()[:, 1])
        stalenesses.append(np.add(data.get_values()[:, staleness_type], 1)) #have to add one because staleness otherwise starts from 0
    if over_all:
        zip_list = []
        last_elem = 0
        for l in stalenesses:
            x_vals, y_vals = np.unique(l, return_counts=True)
            print(x_vals.tolist())
            zip_list.append(list(zip(x_vals.tolist(), y_vals.tolist())))
            lst = x_vals[-1]
            print(lst)
            if last_elem < lst:
                last_elem = lst
        print("last_elem", last_elem)
        y_vals_list = []
        for v in zip_list:
            y_list = []
            last = 1
            print(v)
            for c, (x, y) in enumerate(v):
                while x > last:
                    print(last, 0)
                    y_list.append(0)
                    last += 1
                else:
                    print(x, y)
                    y_list.append(y)
                    last += 1
            if last < last_elem:
                y_list = y_list + range(last + 1, last_elem + 1)
            print(y_list)
            y_vals_list.append(y_list)
        y_dist = np.mean(y_vals_list, axis=0)
        x_dist = list(range(1, last_elem + 1))
        total_updates = np.sum(y_dist)
        y_dist = np.divide(y_dist, total_updates)
        print("y_list",y_dist)
        print("x_list",x_dist)
        plt.subplot(dims[i])
        plt.bar(x_dist, y_dist, width = 1)
        #plt.axis([0,19000,0,0.005])
        plt.xlabel("Staleness", fontsize = fontsize_labels)
        plt.title(titles[i], fontsize = fontsize_titles)

        avg = (np.average(x_dist, weights=y_dist))
        print("average:", avg)
        flat = np.array(stalenesses).flatten()
        median = (np.median(flat))
        print("median:", median)

        std = np.std(flat)
        print("std:", std)

        x = np.arange(1, last_elem, 0.01)
        # plt.plot(x, 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-(x - avg) ** 2 / (2 * std ** 2)))
        # plt.plot(x,np.exp(-(x-avg)**2/(2*std**2)))
plt.tight_layout()
plt.show()
