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
num_reps = 200
rolloutLength = 100
numDemos = [1,3,5,7,9]
tol = 0.0001
gamma = 0.95
burn = 100
skip = 20
delta_conf = 0.95
stochastic = 1
bounds = ["VaR 99","VaR 95", "VaR 90", "WFCB"]
fmts = ['o-','s--','^-.', '*:','>-','d--']


filePath = "data/gridworld/"


print("UPPER BOUND ON SAMPLES IS b =", b)

for bound_type in bounds:
    accuracies = []
    average_bound_error = []
    for numDemo in numDemos:  
        true_perf_ratio = []
        predicted = []
        bound_error = []  
          
        print("=========", numDemo, "=========")
        for rep in range(num_reps):
            filename = "GridWorldInfHorizon_numdemos" + str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) + "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) + "_rolloutLength" + str(rolloutLength) + "_stochastic" + str(stochastic) + "_rep" + str(rep)+ ".txt"
            #print filename
            f = open(filePath + filename,'r')   
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
            elif bound_type == "VaR 99":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.99, delta_conf)
            elif bound_type == "VaR 95":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.95, delta_conf)
            elif bound_type == "VaR 90":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.9, delta_conf)
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
        print(bound_type)
        #print predicted
        print("accuracy", accuracy)
        accuracies.append(accuracy)
        average_bound_error.append(np.mean(bound_error))
        print() 
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

    #plt.savefig("upperBoundError9x9gridNoTerminal_NoDups_alpha" + str(alpha) + ".png")

    fig_cnt = 2
    plt.figure(fig_cnt)
    #plt.title(r"5x5 navigation domain, $\alpha = $" + str(alpha) + ", $step = $" + str(step))
    plt.xlabel("number of demonstrations", fontsize=19)
    plt.ylabel("accuracy", fontsize=19)
    plt.plot(numDemos, accuracies, fmts[bounds.index(bound_type)], label= bound_type, lw = 2)
    #plot 95% confidence line
    #plt.plot([numDemos[0], numDemos[-1]],[0.95, 0.95], 'k--', lw=1)
    plt.xticks(numDemos, fontsize=18) 
    plt.yticks([0.84,0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.001],[0.84, 0.86,0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0,''], fontsize=18)
    plt.legend(loc='lower right',fontsize=19)
    plt.tight_layout()
    #plt.savefig("accuracy9x9gridNoTerminal_expNoDups_alpha" + str(alpha) + ".png") 
    


plt.show()
    

