import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf

#plot results for experiment7_1
#rewards are feasible in that all start states end up at goal within 25 steps

sample_flag = 4
chain_len = 2000
mcmc_step = 0.01
alpha = 10
num_reps = 20
mc_numRollouts = 200
numDemos = [1]
tol = 0.0001
demo_length = 100;
gamma = 0.95
burn = 20 
skip = 2
delta_conf = 0.95
bounds = ["WFCB", "VaR 95"]
fmts = ['o-','s--','^-.', '*:','>-','d--']

filePath = "data/driving/"

evalPolicyNames = ["right_safe","on_road","nasty"]
print("\t\t" + bounds[0] +"\t\t" + bounds[1])

for evalPolicyName in evalPolicyNames:
    data_row = []
    data_row.append(evalPolicyName)
    for bound_type in bounds:
        accuracies = []
        average_bound_error = []
       
        for numDemo in numDemos:  
            true_perf_ratio = []
            predicted = []
            bound_error = []  
            worst_cases = []
            actuals = []
            for rep in range(num_reps):
                filename = "driving_" + evalPolicyName + "_alpha" + str(alpha) + "_chain" + str(chain_len) + "_step" + str(mcmc_step)+ "0000_L1sampleflag" + str(sample_flag) + "_demoLength" + str(demo_length)+ "_mcRollout" + str(mc_numRollouts) + "_rep" + str(rep)+ ".txt";
                

                #print filename
                f = open(filePath + filename,'r')   
                f.readline()                                #clear out comment from buffer
                actual = (float(f.readline())) #get the true ratio 
                actuals.append(actual)
                #print "actual", actual
                f.readline()                                #clear out ---
                wfcb = (float(f.readline())) #get the worst-case feature count bound
                worst_cases.append(wfcb)
                #print wfcb
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
                if bound_type == "VaR 99":
                    upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.99, delta_conf)
                elif bound_type == "VaR 95":
                    upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.95, delta_conf)
                elif bound_type == "VaR 90":
                    upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.9, delta_conf)
                elif bound_type == "WFCB":
                    upper_bnd = wfcb
               
                
                #print "upper bound", upper_bnd
                predicted.append(upper_bnd)
            ave_evd_bound = np.mean(predicted)
            data_row.append(ave_evd_bound)
    print(data_row)
            

