import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf

#plot results for experiment7_1
#rewards are feasible in that all start states end up at goal within 25 steps

sample_flag = 4
chain_length = 10000
step = 0.01
alpha = 100
size = 9
num_reps = 20
rolloutLength = 100
numDemo = 1
tol = 0.0001
gamma = 0.95
burn = 100 #TODO play with this
skip = 20
delta_conf = 0.95
samples = []
filePath = "../../data/enoughIsEnough/experiment_toy/"

for rep in range(num_reps):
    filename = filePath + "numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) +  "_rolloutLength" + str(rolloutLength) + "_stochastic0_rep" + str(rep)+ ".txt"
    #print filename
    f = open(filename,'r')   
    f.readline()                                #clear out comment from buffer
    true_return = (float(f.readline())) #get the true opt policy exp return
    f.readline()
    actual = (float(f.readline())) #get the true ratio 
    #print "actual", actual
    f.readline()                                #clear out ---
    wfcb = (float(f.readline())) #get the worst-case feature count bound
    f.readline()  #clear out ---
    for line in f:                              #read in the mcmc chain
        val = float(line)                       
        samples.append(float(line))
fig_cnt = 1
plt.figure(fig_cnt)
#plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
plt.xlabel("number of demonstrations", fontsize=19)
plt.ylabel("average bound error", fontsize=19)
#plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
plt.hist(samples)
plt.title("step: " + str(step))
#plt.savefig("boundErrorToy_alpha" + str(alpha) + ".png") 
print len(samples)
print max(samples)

alpha = 1
step = 0.05


samples = []
for rep in range(num_reps):
    filename = filePath + "numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) +  "_rolloutLength" + str(rolloutLength) + "_stochastic0_rep" + str(rep)+ ".txt"
    #print filename
    f = open(filename,'r')   
    f.readline()                                #clear out comment from buffer
    true_return = (float(f.readline())) #get the true opt policy exp return
    f.readline()
    actual = (float(f.readline())) #get the true ratio 
    #print "actual", actual
    f.readline()                                #clear out ---
    wfcb = (float(f.readline())) #get the worst-case feature count bound
    f.readline()  #clear out ---
    for line in f:                              #read in the mcmc chain
        val = float(line)                       
        samples.append(float(line))
print len(samples)
print max(samples)
fig_cnt = 2
plt.figure(fig_cnt)
#plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
plt.xlabel("number of demonstrations", fontsize=19)
plt.ylabel("average bound error", fontsize=19)
#plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
plt.hist(samples)
#plt.savefig("boundErrorToy_alpha" + str(alpha) + ".png") 
plt.title("step: " + str(step))

plt.show()
