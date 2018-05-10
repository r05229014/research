import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

dir = '/home/ericakcc/Desktop/ericakcc/research/result/seminar/csv_history/'
name_list = os.listdir(dir)
print(name_list[0])
for name in name_list:
    print(name)
    a = pd.read_csv(dir + name, skiprows=[0], header = None)
    a = a.values
    print(a.shape)
    print(a[:,0])

    fig, ax1 = plt.subplots(figsize=(12,10))
    plt.title(name)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.plot(np.arange(a.shape[0]), a[:,0], 'r--', label ='loss')
    ax1.plot(np.arange(a.shape[0]), a[:,1], 'r', label = 'val_loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(np.arange(a.shape[0]), a[:,2], 'b--', label = 'acc')
    ax2.plot(np.arange(a.shape[0]), a[:,3], 'b', label = 'val_acc')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
