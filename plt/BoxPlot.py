import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib 
file_names = ['Softsync_sync_{0}_log_1_scale=16_#agents=16:itr=0:spcl=False_mom.log',
            'Softsync_async_fast_{0}_log_16_scale=16_#agents=16:itr=0:spcl=False_mom.log',
            'Softsync_async_nomc_{0}_log_16_scale=16_#agents=16:itr=0:spcl=False_mom.log',
            'Softsync_async_{0}_log_16_scale=16_#agents=16:itr=0:spcl=False_mom.log',
            'Softsync_async_stepwise_{0}_log_16_scale=1_#agents=16:itr=0:spcl=True_mom.log'
            ]

# List a color for Agent we want to plot
colours = ["black", "green", "red", "orange","yellow","pink","blue","grey"] #can also just give range(0, number_of_colours), if we set auto_colorus = True
# if true use default seaborn colours
auto_colours = False
# number of epochs
epochs = 100
#if there is more than one evaluation of the metrics per epoch
evals_per_epoch = 1
# how many times the simulator was rerun with the same parameters.
repeats = 5
#column in the csv file that coresponds to named metric
indexes = {'Train Loss':2, 'Train Accuracy':3, 'Test Loss':5, 'Test Accuracy':6}

names = ["Sync","Async RS+MC", "Async RS", "Async RS+MC \n slow", "Async SwStl \n slow"]


test_accuracy_max = []

for name in file_names:
    res3_max = []
    for j in range(0, repeats):
        #change the file name here
        data = pd.read_csv(name.format(j+1))
        res = data.get_values()[:, indexes['Test Accuracy']]
        res3_max.append(np.max(res))


    test_accuracy_max.append(res3_max)

plt.figure(figsize=[15,5])
plots = [131,132,133]


for i, ta in enumerate(test_accuracy_max):
    #plt.subplot(plots[i])
    print(ta)
    print(names[i],"maximum is",np.max(ta),"std is", np.std(ta), flush=True)
plt.boxplot(test_accuracy_max, notch =True, whis = 1000, labels = names)

plt.xticks(fontsize=20)
plt.ylabel("Maximum Accuracy", fontsize = 20)

#plt.title(names[i])

plt.show()


