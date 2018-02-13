import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf

#plot results for experiment2_3
#rewards are sparse only three non-zero, counting the terminal at 0.5.
#plot the errors for my simple example of feature counts

sample_flag = 4
chain_length = 5000
step = 0.01
alpha = 50
size = 5
num_reps = 100
numDemos = [1,2,3,4]
tol = 0.0001
gamma = 0.95
burn = 100 #TODO play with this
skip = 10
delta_conf = 0.95
bounds = ["PB 99", "PB 95", "PB 90", "WFCB", "CH", "MPeB"]
fmts = ['o-','s--','^-.', '*:','>-','d--']
b = 2 * 1.0/(1.0 - gamma)  #upper bound on random variables for concentration inequalities
c = 10 #truncation value for MPeBCollapsed (Phil's method)

filePath = "/home/daniel/Code/PeARL_BIRL/data/experiment2_3/"

print "UPPER BOUND ON SAMPLES IS b =", b

for bound_type in bounds:
    accuracies = []
    average_bound_error = []
    for numDemo in numDemos:  
        true_perf_ratio = []
        predicted = []
        bound_error = []  
          
        print "=========", numDemo, "========="
        for rep in range(num_reps):
            #print "rep:",rep
            if(step < 0.1):
                filename = filePath + "numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) + "_rep" + str(rep)+ ".txt"
            else:
                filename = filePath + "numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "00000_L1sampleflag" + str(sample_flag) + "_rep" + str(rep)+ ".txt"
            #print filename
            f = open(filename,'r')   
            f.readline()                                #clear out comment from buffer
            actual = (float(f.readline())) #get the true ratio 
            #print "actual", actual
            f.readline()                                #clear out ---
            wfcb = (float(f.readline())) #get the worst-case feature count bound
            f.readline()  #clear out ---
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
            elif bound_type == "PB 99":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.99, delta_conf)
            elif bound_type == "PB 95":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.95, delta_conf)
            elif bound_type == "PB 90":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.9, delta_conf)
            elif bound_type == "aveVal":
                upper_bnd = np.mean(burned_samples)
            elif bound_type == "AM":
                upper_bnd = -1*bound_methods.anderson_lower_bnd(-np.array(burned_samples), delta_conf)
            elif bound_type == "MPeB":
                upper_bnd = bound_methods.empirical_bernstein_bnd(burned_samples, delta_conf, b)
            elif bound_type == "WFCB":
                upper_bnd = wfcb
                
            
            
            #print "upper bound", upper_bnd
            predicted.append(upper_bnd)
            true_perf_ratio.append(actual)
            bound_error.append(upper_bnd - actual)
        accuracy = 0.0
        for i in range(len(predicted)):
            if (predicted[i] >= true_perf_ratio[i]) or np.abs(predicted[i] - true_perf_ratio[i]) < tol:
                accuracy += 1.0
        accuracy = accuracy / len(predicted)
        print bound_type
        #print predicted
        print "accuracy", accuracy
        accuracies.append(accuracy)
        average_bound_error.append(np.mean(bound_error))
        print 
    fig_cnt = 1
    plt.figure(fig_cnt)
    #plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
    plt.xlabel("number of demonstrations", fontsize=19)
    plt.ylabel("average bound error", fontsize=19)
    #plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
    plt.plot(numDemos, average_bound_error, fmts[bounds.index(bound_type)], label=bound_type, lw=2)
    #plot dotted line across diagonal
    #plt.plot([0,true_perf_ratio[-1]],[0,true_perf_ratio[-1]], 'k:')
    plt.xticks(numDemos,fontsize=18) 
    plt.yticks(fontsize=18) 
    plt.legend(loc='best', fontsize=19)
    plt.tight_layout()

    plt.savefig("upperBoundError9x9gridgoal_exp2_3.png")

    fig_cnt = 2
    plt.figure(fig_cnt)
    #plt.title(r"5x5 navigation domain, $\alpha = $" + str(alpha) + ", $step = $" + str(step))
    plt.xlabel("number of demonstrations", fontsize=19)
    plt.ylabel("accuracy", fontsize=19)
    plt.plot(numDemos, accuracies, fmts[bounds.index(bound_type)], label= bound_type, lw = 2)
    #plot 95% confidence line
    plt.plot([numDemos[0], numDemos[-1]],[0.95, 0.95], 'k--', lw=1)
    plt.xticks(numDemos, fontsize=18) 
    plt.yticks([0.9, 0.925, 0.95, 0.975, 1.0, 1.001],[0.9, 0.925, 0.95, 0.975, 1.0,''], fontsize=18)
    plt.legend(loc='best',fontsize=19)
    plt.tight_layout()
    plt.savefig("accuracy9x9gridgoal_exp2_3.png") 
    


plt.show()
    
##code to plot distribution over estiamted value differences
#all_samples = []
#for rep in range(num_reps):
##print rep
#filename = "/home/daniel/Code/PeARL_BIRL/data/fcountToyMLE/fcount_badeval_alpha" + str(alpha) + "_chain" + str(num_mcmc_samples) + "_L1sampleflag" + str(sample_flag) + "_steps" +str(num_steps) + "_rep" + str(rep)+ ".txt";
##print filename
#f = open(filename,'r')   
#f.readline()                                #clear out comment from buffer
#actual = (float(f.readline())) #get the true ratio 
##print "actual", actual
#f.readline()                                #clear out ---
#samples = []
#for line in f:                              #read in the mcmc chain
#    val = float(line)                       
#    samples.append(float(line))
##print samples
##burn 
#burned_samples = samples[burn::skip]
#all_samples.extend(burned_samples)
#print "max val", max(all_samples)
#print "min val", min(all_samples)
#print "ave val", np.mean(all_samples)
#print "median val", np.median(all_samples)
#plt.figure(1)
#plt.xticks(fontsize=16)  
#plt.yticks(fontsize=16)  
#plt.hist(all_samples,bins=60)
#plt.title(r"Distribution over MCMC chain for $\pi_2$", fontsize = 18)
#plt.xlabel("Absolute Value Difference ",fontsize=18)
#plt.ylabel("frequency",fontsize=18)
#plt.tight_layout()
#plt.legend(loc='best')
#plt.savefig("/home/daniel/Documents/ScottResearch/SafeIRL/samplingL1UnitBall/UnitSphereSampling/badpolicyValueDiffDist.png")
#plt.show()
