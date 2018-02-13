import numpy as np
import matplotlib.pyplot as plt
from numpy import nan, inf

filePath = "/home/dsbrown/Code/PeARL_BIRL/data/machine_teaching/"

filename = "cakmak_10000_task" 


def to_table_row(data_array):
    row = ""
    for i in range(len(data_array)):
        if i < (len(data_array) - 1):
            separator = " & "
        else:
            separator = " \\\\"
            
        if np.round(data_array[i]) == data_array[i]:
            row += '{0:.1f}'.format(data_array[i]) + separator
        else:
            row += '{0:.3f}'.format(data_array[i]) + separator
    return row


all_data = []
for task in range(1,5):
    f = open(filePath + filename + str(task) + ".txt",'r')
    times = []
    demos = []
    losses = []
    for line in f:
        data = line.strip().split(",")
        times.append(float(data[0]))
        demos.append(float(data[1]))
        losses.append(float(data[2]))
    all_data.append(np.mean(demos))    
    all_data.append(np.mean(losses))
    all_data.append(np.mean(times))
print all_data
print to_table_row(all_data)
        

