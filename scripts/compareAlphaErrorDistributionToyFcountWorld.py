import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf
#code to plot distribution over estiamted value differences


sample_flag = 4
num_steps = 10
num_mcmc_samples = 10000
alpha = 10
size = 4
num_reps = 100
tol = 0.0001
burn = 1000 #TODO play with this
skip = 1
filenames = []
alphas = [1,10,100]

alpha_samples = []
for a in alphas:
    all_samples = []
    for rep in range(num_reps):
        #print rep
        #print filename
        fname = "/home/daniel/Code/PeARL_BIRL/data/fcountAlpha" + str(a) + "/fcount_badeval_alpha" + str(a) + "_chain" + str(num_mcmc_samples) + "_L1sampleflag" + str(sample_flag) + "_steps" +str(num_steps) + "_rep" + str(rep)+ ".txt"
        f = open(fname,'r')   
        f.readline()                                #clear out comment from buffer
        actual = (float(f.readline())) #get the true ratio 
        #print "actual", actual
        f.readline()                                #clear out ---
        samples = []
        for line in f:                              #read in the mcmc chain
            val = float(line)                       
            samples.append(float(line))
        #print samples
        #burn 
        burned_samples = samples[burn::skip]
        all_samples.extend(burned_samples)
    alpha_samples.append(all_samples)
    print "-----alpha", a, "----"
    print "max val", max(all_samples)
    print "min val", min(all_samples)
    print "ave val", np.mean(all_samples)
    print "median val", np.median(all_samples)
plt.figure(1)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)  
data_to_plot = np.vstack([samps for samps in alpha_samples]).T
plt.hist(data_to_plot,bins=30, label=[str(a) for a in alphas])
plt.title(r"Distribution over MCMC chain for $\pi_2$", fontsize = 18)
plt.xlabel("Absolute Value Difference ",fontsize=18)
plt.ylabel("frequency",fontsize=18)
plt.tight_layout()
plt.legend(loc='best')
plt.savefig("/home/daniel/Documents/ScottResearch/SafeIRL/samplingL1UnitBall/UnitSphereSampling/badpolicyValueDiffDist.png")
plt.show()
