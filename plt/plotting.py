import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

file_names = [["Softsync_s_{0}_log_1_scale=8_#agents=8:itr=0:spcl=False_mom.log",
         "Softsync_mcasS_{0}_log_8_scale=8_#agents=8:itr=0:spcl=False_mom.log",
         "Softsync_mcasSawr_{0}_log_8_scale=1_#agents=8:itr=0:spcl=True_mom.log",
         "Softsync_mcasSawr2_{0}_log_8_scale=1_#agents=8:itr=0:spcl=True_mom.log",
         "Softsync_mcasSbin_{0}_log_8_scale=8_#agents=8:itr=0:spcl=False_mom.log"]

# List a color for Agent we want to plot
colours = ["black", "blue", "red", ]#"orange","yellow","pink","blue","grey"] #can also just give range(0, number_of_colours), if we set auto_colorus = True
# if true use default seaborn colours
auto_colours = False
# number of epochs
epochs = 100
#if there is more than one evaluation of the metrics per epoch
evals_per_epoch = 1
# how many times the simulator was rerun with the same parameters.
repeats = 5
#column in the csv file that coresponds to named metric
indexes = {'Train Loss':2, 'Train Accuracy':3, 'Test Loss':4, 'Test Accuracy':5}

names = ["Baseline", "Asynchronous 8 agents", "Asynchronous 8 agents, scaled learning rate"]
fontsize_titles=14
fontsize_labels = 12


#List to store the read data points
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for name in file_names:
    res2 = []
    res3 = []
    res0 = []
    res1 = []
    for j in range(1, repeats+1):
        #change the file name here
        data = pd.read_csv(name.format(j), header = None)
        res0.append(data.get_values()[:, indexes['Train Loss']])
        res1.append(data.get_values()[:, indexes['Train Accuracy']])
        res2.append(data.get_values()[:, indexes['Test Loss']])
        res3.append(data.get_values()[:, indexes['Test Accuracy']])
    train_loss.append(res0)
    train_accuracy.append(res1)
    test_loss.append(res2)
    test_accuracy.append(res3)

plt.figure(figsize=[15,10])
plots = [221,222,223,224]
metrics_ensemble = [train_loss,train_accuracy,test_loss,test_accuracy]

nr_agents = len(file_names)
stride = epochs*evals_per_epoch

for key, p in enumerate(plots):
    plt.subplot(p)

    x = np.zeros([repeats * nr_agents * stride])
    for i in range(0, nr_agents * repeats):
        x[i * stride:(i + 1) * stride] = np.arange(1./evals_per_epoch, epochs + 1./evals_per_epoch, 1./evals_per_epoch)

    h = []
    for c, i in enumerate(colours):
        for j in range(repeats * stride):
            h.append(c)

    y = np.zeros([repeats * nr_agents * stride])
    for i, d in enumerate(metrics_ensemble[key]):
        for j, dd in enumerate(d):
            y[i * stride * repeats + j * stride:i * stride * repeats + (j + 1) * stride] = dd

    plt.title(list(indexes.keys())[key], fontsize=fontsize_titles)
    if auto_colours:
        plt_sb = sb.lineplot(x=x, y=y, hue=h, ci="sd")
    else:
        plt_sb = sb.lineplot(x=x, y=y, hue=h, palette=colours, ci="sd")
    agent = names
    plt.legend(agent, loc="center right",fontsize = fontsize_labels)
    plt.xlabel("Epochs", fontsize = fontsize_labels)
    #plt.ylabel("Accuracy", fontsize = fontsize_labels)
plt.tight_layout()
plt.show()

