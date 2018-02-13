import bound_methods
import numpy as np
import matplotlib.pyplot as plt


bounds = ["VAR", "PB", "MPeBC", "CH"]
burn = 100
skip = 1
delta_conf = 0.95
percentile = 0.95
filename = "/home/daniel/Code/PeARL_BIRL/test.txt"
#print filename
f = open(filename,'r')   
f.readline()                                #clear out comment from buffer
actual = (float(f.readline()))              #get the true ratio 
#print "actual", actual
f.readline()                                #clear out ---
samples = []
for line in f:                              #read in the mcmc chain
    val = float(line)                       
    samples.append(float(line))
#print samples
#burn 
final_samples = samples[burn::skip]
print "using", len(final_samples), "samples"
print "average of samples", np.mean(final_samples)
print "true bound", actual
for bound_type in bounds:
    #predicted = []
    #compute confidence bound
    if bound_type == "VAR":
        upper_bnd = bound_methods.value_at_risk(final_samples, delta_conf)
    elif bound_type == "MPeBC":
        upper_bnd = bound_methods.phil_upper_bnd(final_samples, delta_conf, 40)
    elif bound_type == "TT":
        upper_bnd = bound_methods.ttest_upper_bnd(final_samples, delta_conf)
    elif bound_type == "BS":                   
        upper_bnd = bound_methods.bootstrap_percentile_confidence_upper(final_samples, delta_conf, num_bootstrap)
    elif bound_type == "CH":
        upper_bnd = bound_methods.chernoff_hoeffding_upper_bnd(final_samples, delta_conf, 100)
    elif bound_type == "PB":
        upper_bnd = bound_methods.percentile_confidence_upper_bnd(final_samples, percentile, delta_conf)

    print bound_type
    print "upper bound", upper_bnd
    #predicted.append(upper_bnd)
plt.plot(final_samples)
plt.show()

