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
features = range(2,51)
N = 5
for F in features:
    print("---",N)
    all_data = []
    f = open(filePath + "non-redundant_N=" + str(N) + "_F=" + str(F) + "_random_trajLength=1.txt",'r')
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
plt.fill_between(features, num_demos - demo_stdev, num_demos + demo_stdev,facecolor='blue', interpolate=True, alpha=0.2)
plt.plot(features,num_demos,'b-',lw=2,label="binary")

num_demos = []
demo_stdev = []
for F in features:
    print("---",N)
    all_data = []
    f = open(filePath + "non-redundant_N=" + str(N) + "_F=" + str(F) + "_random_mixed_trajLength=1.txt",'r')
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
plt.fill_between(features, num_demos - demo_stdev, num_demos + demo_stdev,facecolor='green', interpolate=True, alpha=0.2)
plt.plot(features,num_demos,'g-',lw=2,label="random")


plt.ylabel("$|\mathcal{D}|$", fontsize=32)
plt.xlabel("Features", fontsize=32)
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30) 
plt.legend(loc='best',fontsize=32)
plt.tight_layout()
#plt.savefig("accuracy9x9NoisyTgridNoTerminal_expNoDups_alpha" + str(alpha) + ".png") 
plt.show()

        

