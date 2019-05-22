import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time as tm
import math
# Log_softsync1_scale=1_#agents=4:itr=1
array = np.zeros((180,4))
"""
file_names = ['Log_softsync16_scale=16_#agents=16:itr=0:exp=Falsebinned.log',
'Log_softsync16_scale=16_#agents=16:itr=0:exp=Falsecorr.log',
'Log_softsync16_scale=16_#agents=16:itr=0:exp=False.log',
'Log_softsync16_scale=16_#agents=16:itr=0:spcl=Falsestl_corr.log',
'Log_softsync16_scale=16_#agents=16:itr=0:spcl=Falsestl.log',
'Log_softsync8_scale=1_#agents=8:itr=0:spcl=False.log',
'Log_softsync8_scale=8_#agents=8:itr=0:exp=True.log',
'Log_softsync8_scale=8_#agents=8:itr=0:spcl=False_bin_slow.log',
'Log_softsync8_scale=8_#agents=8:itr=0:spcl=False_cor_slow.log',
'Log_softsync8_scale=8_#agents=8:itr=0:spcl=Falsestl_corr_half.log',
'Log_softsync8_scale=8_#agents=8:itr=0:spcl=Falsestl_corr.log',
'Log_softsync8_scale=8_#agents=8:itr=0:spcl=Falsestl.log',
]"""
file_names = ['Softsync_accrued_p=01__log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log',
'Softsync_accrued_p=05__log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log',
'Softsync_accrued_p=09__log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log',
'Softsync_accrued_std=01__log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log',
'Softsync_p=01__log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log',
'Softsync_p=05__log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log',
'Softsync_p=09__log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log',
'Softsync_std=01_log_8_scale=8_#agents=8:itr=0:spcl=False_acc.log']

for name in file_names:
    maxlist = []
    data = pd.read_csv(name)
    maxlist.append(data.get_values()[:, 5])
    print(np.max(maxlist))
"""
for i in [2,4]:
    maxlist = []
    for j in range(0,5):
        data = pd.read_csv("Log_"+str(i)+"_agent_h_mnist_itr:"+str(j)+".log")
        maxlist.append(data.get_values()[:, 4])
    print(np.max(maxlist))"""