import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf

#plot results for experiment4_1
#rewards are feasible in that all start states end up at goal within 25 steps

sample_flag = 4
chain_length = 10000
step = 0.01
alpha = 100
size = 9
num_reps = 200
rolloutLength = 100
numDemos = [1,5,9]
tol = 0.0001
gamma = 0.95
burn = 100 #TODO play with this
skip = 20
stochastic = 1
delta_conf = 0.95
bounds = ["VaR 95", "VaR 99"]
bound_dict = {}

filePath = "./data/abbeel_projection/"


for bound_type in bounds:
    accuracies = []
    average_bound = []
    for numDemo in numDemos:  
        predicted = []
        bound_error = []  
        true_perf_ratio = []
        #print("=========", numDemo, "=========")
        for rep in range(num_reps):
            filename = "ProjectionEval_numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) +  "_rolloutLength" + str(rolloutLength) +  "_stochastic" + str(stochastic) + "_rep" + str(rep)+ ".txt"
                
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
            if bound_type == "VaR 99":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.99, delta_conf)
            elif bound_type == "VaR 95":
                upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.95, delta_conf)

                
            
            
            
            #print "upper bound", upper_bnd
            predicted.append(upper_bnd)
            true_perf_ratio.append(actual)
            bound_error.append(upper_bnd - actual)
        accuracy = 0.0
        for i in range(len(predicted)):
            if (predicted[i] >= true_perf_ratio[i]) or np.abs(predicted[i] - true_perf_ratio[i]) < tol:
                accuracy += 1.0
        accuracy = accuracy / len(predicted)
     
        accuracies.append(accuracy)
        average_bound.append(np.mean(predicted))
    bound_dict[bound_type] = average_bound
    print(bound_type)
    print("over", numDemos, "demos")
    print("bound", average_bound)
    print("ave accuracy", np.mean(accuracies))

#figure out Syed and Schapire theoretical bounds to compare against
VaR95_bound = bound_dict["VaR 95"][0]
syed_bounds = []
k = 8 #num_features


for ndemo in numDemos:
    syed_bounds.append(bound_methods.syed_schapire_bound(ndemo, gamma, k, delta_conf))

#calculate Syed and Schapire bound to match our bound with 1 demo
m = 0
eps = 100000
while(eps > VaR95_bound):
    m = m + 1
    #print("count", m)
    eps = bound_methods.syed_schapire_bound(m, gamma, k, delta_conf)
    #print('syed', eps1)
syed_bounds.append(eps)
numDemos.append(m)
print("Syed and Schapire")
print("over", numDemos, "demos")
print("bound", syed_bounds)

