import numpy as np
import matplotlib.pyplot as plt
from numpy import nan, inf

filePath = "/home/dsbrown/Code/PeARL_BIRL/data/machine_teaching/"




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

num_demos = []
demo_stdev = []
world_sizes = range(3,31)
for N in world_sizes:
    print("---",N)
    all_data = []
    f = open(filePath + "non-redundant_N=" + str(N) + "_f=8_random_trajLength=1.txt",'r')
    times = []
    demos = []
    losses = []
    zero_one_loss = []
    num_sa_pairs = []
    for line in f:
        data = line.strip().split(",")
        times.append(float(data[0]))
        demos.append(float(data[1]))
        num_sa_pairs.append(float(data[2]))
    print(np.mean(num_sa_pairs))
    print(np.mean(times))
    num_demos.append(np.mean(num_sa_pairs))
    demo_stdev.append(np.std(num_sa_pairs))
num_demos = np.array(num_demos)
demo_stdev = np.array(demo_stdev)

print(num_demos)
print(demo_stdev)
    
plt.figure(1)
plt.fill_between(world_sizes, num_demos - demo_stdev, num_demos + demo_stdev,facecolor='blue', interpolate=True, alpha=0.2)
plt.plot(world_sizes,num_demos,'b-',lw=2,label="binary")


num_demos = []
demo_stdev = []
world_sizes = range(3,31)
for N in world_sizes:
    print("---",N)
    all_data = []
    f = open(filePath + "non-redundant_N=" + str(N) + "_f=8_random_mixed_trajLength=1.txt",'r')
    times = []
    demos = []
    losses = []
    zero_one_loss = []
    num_sa_pairs = []
    for line in f:
        data = line.strip().split(",")
        times.append(float(data[0]))
        demos.append(float(data[1]))
        num_sa_pairs.append(float(data[2]))
    print(np.mean(num_sa_pairs))
    print(np.mean(times))
    num_demos.append(np.mean(num_sa_pairs))
    demo_stdev.append(np.std(num_sa_pairs))
num_demos = np.array(num_demos)
demo_stdev = np.array(demo_stdev)

print(num_demos)
print(demo_stdev)
    
plt.figure(1)
plt.fill_between(world_sizes, num_demos - demo_stdev, num_demos + demo_stdev,facecolor='green', interpolate=True, alpha=0.2)
plt.plot(world_sizes,num_demos,'g-',lw=2,label="random")





plt.legend(loc='lower right',fontsize=32)
plt.axis([3,30,0,35])
plt.ylabel("$|\mathcal{D}|$", fontsize=32)
plt.xlabel("Grid width", fontsize=32)
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30) 
plt.tight_layout()
plt.show()

        

