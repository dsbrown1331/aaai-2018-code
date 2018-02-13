import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf

#plot the errors for my simple example of feature counts

sample_flag = 4
num_steps = 10
num_mcmc_samples = 10000
alpha = 10
size = 4
num_reps = 100
tol = 0.0001
burn = 500 #TODO play with this
skip = 1
delta_conf = 0.95
percentiles = []#[0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
bounds = []#["CH", "TT", "MPeB", "AM"]
fmts = ['o-','s--','^-.', '*:']
b = 100 #upper bound on random variables for concentration inequalities
c = 10 #truncation value for MPeBCollapsed (Phil's method)



print "UPPER BOUND ON SAMPLES IS b =", b

for p in percentiles:
    print "percentile", p
    for bound_type in bounds:
        accuracies = []
        average_bounds = []
        stdev_bounds = []
        true_perf_ratio = []
        predicted = []
        for rep in range(num_reps):
            #print rep
            filename = "/home/daniel/Code/PeARL_BIRL/data/fcountToyMLE/fcount_badeval_alpha" + str(alpha) + "_chain" + str(num_mcmc_samples) + "_L1sampleflag" + str(sample_flag) + "_steps" +str(num_steps) + "_rep" + str(rep)+ ".txt";
            #print filename
            f = open(filename,'r')   
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
            #print "max sample", np.max(burned_samples)
            #compute confidence bound
            if bound_type == "VAR":
                upper_bnd = bound_methods.value_at_risk(burned_samples, delta_conf)
            elif bound_type == "MPeBC":
                upper_bnd = bound_methods.phil_upper_bnd(burned_samples, delta_conf, c)
            elif bound_type == "TT":
                upper_bnd = bound_methods.ttest_upper_bnd(burned_samples, delta_conf)
            elif bound_type == "BS":                   
                upper_bnd = bound_methods.bootstrap_percentile_confidence_upper(burned_samples, delta_conf, num_bootstrap)
            elif bound_type == "CH":
                upper_bnd = bound_methods.chernoff_hoeffding_upper_bnd(burned_samples, delta_conf, b)
            elif bound_type == "PB":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, p, delta_conf)
            elif bound_type == "aveVal":
                upper_bnd = np.mean(burned_samples)
            elif bound_type == "AM":
                upper_bnd = -1*bound_methods.anderson_lower_bnd(-np.array(burned_samples), delta_conf)
            elif bound_type == "MPeB":
                upper_bnd = bound_methods.empirical_bernstein_bnd(burned_samples, delta_conf, b)
                
            
            
            #print "upper bound", upper_bnd
            predicted.append(upper_bnd)
        accuracy = 0.0
        for i in range(len(predicted)):
            if (predicted[i] >= actual) or np.abs(predicted[i] - actual) < tol:
                accuracy += 1.0
        accuracy = accuracy / len(predicted)
        print bound_type
        #print predicted
        print "accuracy", accuracy
        accuracies.append(accuracy)
        average_bounds.append(np.nanmean(predicted))
        true_perf_ratio.append(actual)
        print "true bound", actual
        print "ave bound", np.nanmean(predicted)
        print "stdev bound", np.nanstd(predicted)
        stdev_bounds.append(np.nanstd(predicted))
        print 
#code to plot distribution over estiamted value differences
all_samples = []
for rep in range(num_reps):
    #print rep
    filename = "/home/daniel/Code/PeARL_BIRL/data/fcountToyMLE/fcount_badeval_alpha" + str(alpha) + "_chain" + str(num_mcmc_samples) + "_L1sampleflag" + str(sample_flag) + "_steps" +str(num_steps) + "_rep" + str(rep)+ ".txt";
    #print filename
    f = open(filename,'r')   
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
print "max val", max(all_samples)
print "min val", min(all_samples)
print "ave val", np.mean(all_samples)
print "median val", np.median(all_samples)
plt.figure(1)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)  
plt.hist(all_samples,bins=60)
plt.title(r"Distribution over MCMC chain for $\pi_2$", fontsize = 18)
plt.xlabel("Absolute Value Difference ",fontsize=18)
plt.ylabel("frequency",fontsize=18)
plt.tight_layout()
plt.legend(loc='best')
plt.savefig("/home/daniel/Documents/ScottResearch/SafeIRL/samplingL1UnitBall/UnitSphereSampling/badpolicyValueDiffDist.png")
plt.show()
