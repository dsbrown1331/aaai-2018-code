# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:17:18 2017
Empirical Bootstrap method
@author: daniel
"""

#given a list of elements, compute mean and sample with replacement multiple
#times to get a bunch of sample deviations from mean

import numpy as np
from scipy import stats
from numpy import NaN


def percentile(sample_data, p):
    data_sorted = np.sort(sample_data)
    if p > 0.5:
        idx = np.ceil(len(data_sorted) * p)
    else:
        idx = np.floor(len(data_sorted) * p)
    return data_sorted[idx]
    
def syed_schapire_bound(m, gamma, k, delta):
    return 3.0/(1.0-gamma) * np.sqrt(2.0/m * np.log(2.0*k/delta))
    
def percentile_confidence_upper_bnd(sample_data, percentile, delta_conf):
    """percentile should be a decimal, e.g. 75th percentile is 0.75, delta confidence is the true confidence, e.g. 0.95"""
    
    #sort the data
    data_sorted = np.sort(sample_data)
    
    p = percentile 
    #confidence level
    alpha = 1 - delta_conf
    num_samples = len(data_sorted)
    #print p * num_samples
    bin_mean = num_samples * p
    bin_var = num_samples * p * (1 - p)
    bin_std = np.sqrt(bin_var)
    
    #print "upper bound"
    z_upper = stats.norm.ppf(1-alpha)
    #print z_upper
    upper_order_idx = int(np.ceil(z_upper * bin_std + bin_mean - 0.5))
    #print upper_order_idx
    
    #double check math 
    #print "conf level", stats.norm.cdf((upper_order_idx - 0.5 - bin_mean)/bin_std)
    return data_sorted[upper_order_idx-1] #-1 since zero indexing

#testing scripts for bounds
def main():
    #sample_data = [NaN, 1, 2, 3, NaN,1,1,2,3,4,3,2,1,1,1,1,1]
    #sample_data = np.array([1,2,3,2,1,2,1,2,1,1,1,1,1,100,100,1,1,2,4,5,6,7])
    #sample_data = np.array(np.random.rand(3000))
    #num_b = 10000
    #delta_conf = 0.95
#    print "sample mean", np.nanmean(sample_data)
#    print "boot interval", bootstrap_confidence(sample_data, delta_conf, num_b)
#    print "boot upper", bootstrap_empirical_confidence_upper(sample_data, delta_conf, num_b)
#    print "boot upper percent", bootstrap_percentile_confidence_upper(sample_data, delta_conf, num_b)
#    print "value at risk 95%", value_at_risk(sample_data, delta_conf) 
#    print "phil lower bound 95%", phil_lower_bnd(sample_data, delta_conf, 3)
#    print "phil upper bound 95%", phil_upper_bnd(sample_data, delta_conf, 3)
#    print "t-test", ttest_upper_bnd(sample_data, delta_conf)
    num_features = 8
    gamma = 0.9
    delta = 0.05
    for num_demos in range(10,15100):
        print("abbeel bound for", num_demos, "demos =", abbeel_bound(num_features, gamma, delta, num_demos))

if __name__=="__main__":
    main()
