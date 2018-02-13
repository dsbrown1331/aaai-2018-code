import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf

#plot errors in bounds and accuracy of bounds for different types of sampling strategies
#TODO get a script that allows me to plot over different runs based on a template of the filename and the variables so I could just generate plots over different parameters automatically!

sample_flag = 4
mcmc_samples = [5000]
alpha = 100
size = 5
num_reps = 100
tol = 0.0001
burn = 100 #
delta_conf = 0.95
percentile = 0.95
num_bootstrap = 1000
bounds = ["PB", "CH", "]
fmts = ['o-','s--','^-.', '*:']

#loss_ranges = {1:[3,4,5,6,7,8,9,10,11,12,13,14,15], 2:[3,4,5,6,7,8,9,10,11,12,13,14,15], 4:[3,4,5,6,7,8,9,10,11,12,13,14,15], 8:[3,4,5,6,7,8,9,10,11,12,13,14,15], 25:[3,5,6,7,8,9,10,11,12,13,14,15]}

for num_demos in [1,2,4,8,25]:
    for num_mcmc_samples in mcmc_samples:
        for bound_type in bounds:
            accuracies = []
            average_bounds = []
            stdev_bounds = []
            true_perf_ratio = []
            for politer_steps in [3,2,1,0]: #TODO try 4
                predicted = []
                for rep in range(num_reps):
                    #print rep
                    filename = "/home/daniel/Code/PeARL_BIRL/data/gridNavToy_PIsteps" + str(politer_steps) + "_numdemos" +  str(num_demos)  + "_alpha" + str(alpha) + "_chain" + str(num_mcmc_samples) + "_L1sampleflag" + str(sample_flag) + "_rep" + str(rep)+ ".txt";
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
                    burned_samples = samples[burn:]
                    #compute confidence bound
                    if bound_type == "VAR":
                        upper_bnd = bound_methods.value_at_risk(burned_samples, delta_conf)
                    elif bound_type == "MPeBC":
                        upper_bnd = bound_methods.phil_upper_bnd(burned_samples, delta_conf, 40)
                    elif bound_type == "TT":
                        upper_bnd = bound_methods.ttest_upper_bnd(burned_samples, delta_conf)
                    elif bound_type == "BS":                   
                        upper_bnd = bound_methods.bootstrap_percentile_confidence_upper(burned_samples, delta_conf, num_bootstrap)
                    elif bound_type == "CH":
                        upper_bnd = bound_methods.chernoff_hoeffding_upper_bnd(burned_samples, delta_conf, 100)
                    elif bound_type == "PB":
                        upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, percentile, delta_conf)
                    
                    
                    #print "upper bound", upper_bnd
                    predicted.append(upper_bnd)
                accuracy = 0.0
                for i in range(len(predicted)):
                    if (predicted[i] >= actual) or np.abs(predicted[i] - actual) < tol:
                        accuracy += 1.0
                accuracy = accuracy / len(predicted)
                print bound_type
                #print "politer steps", politer_steps, "accuracy", accuracy
                accuracies.append(accuracy)
                average_bounds.append(np.nanmean(predicted))
                true_perf_ratio.append(actual)
                #print "true bound", actual
                #print "predicted bounds", predicted
                stdev_bounds.append(np.nanstd(predicted))
                #print 
        
            fig_cnt = 1
            plt.figure(fig_cnt)
            plt.title(str(num_demos) + " Demos " + r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $N = $" + str(num_mcmc_samples))
            plt.xlabel("true performance ratio", fontsize=18)
            plt.ylabel("average upper bound", fontsize=18)
            plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
            #plt.plot(true_perf_ratio, average_bounds, fmts[bounds.index(bound_type)], label=bound_type, lw=1)
            #plot dotted line across diagonal
            plt.plot([0,true_perf_ratio[-1]],[0,true_perf_ratio[-1]], 'k:')
            plt.xticks(fontsize=15) 
            plt.yticks(fontsize=15) 
            plt.legend(loc='best', fontsize=18)
        
            #    plt.savefig("lower_bound_accuracy_95conf_mcmc_exp6.png")
        
            fig_cnt = 2
            plt.figure(fig_cnt)
            plt.title(str(num_demos) + " Demos " + r"5x5 navigation domain, $\alpha = $" + str(alpha) + ", $N = $" + str(num_mcmc_samples))
            plt.xlabel("true performance ratio", fontsize=17)
            plt.ylabel("accuracy", fontsize=17)
            plt.plot(true_perf_ratio, accuracies, 'o-',label= bound_type, lw = 1)
            #plot 95% confidence line
            plt.plot([0,true_perf_ratio[-1]],[0.95, 0.95], 'k:')
            plt.xticks(fontsize=15) 
            plt.yticks(fontsize=15)
            plt.legend(loc='best',fontsize=18)
            #    plt.savefig("true_return_ratio_mcmc_exp6.png")
    plt.figure(1)
    #plt.savefig("policyupperbound5x5alpha100_qtest_D" + str(num_demos) + ".png")
    plt.figure(2)
    #plt.savefig("policyaccuracy5x5alpha100_qtest_D" + str(num_demos) + ".png")
    plt.show()
        
