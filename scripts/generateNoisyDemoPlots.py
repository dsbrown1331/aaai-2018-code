import numpy as np
import matplotlib.pyplot as plt
import bound_methods
from numpy import nan, inf
from matplotlib.pyplot import cm 

#plot results for experiment4_1
#rewards are feasible in that all start states end up at goal within 25 steps
bar_width = 0.15
opacity = 0.8


sample_flag = 4
chain_length = 10000
step = 0.01
alphas = [0,1,5,10,50]
size = 9
num_reps = 200
rolloutLength = 100
stochastic = 1
randActionProb = 0.2
numDemos = [1,5,9]
tol = 0.0001
gamma = 0.9
burn = 100 
skip = 20
delta_conf = 0.95
bounds = ["VaR 95"]
fmts = ['o-','s--','^-.', '*:','>-','d--']

filePath = "./data/gridworld_noisydemo_exp/"

color=iter(cm.rainbow(np.linspace(0,1,6)))

alpha = 5
index = np.arange(len(numDemos))
cnt = 0;
##########PLOT WFCB
bound_type = "WFCB"
accuracies = []
average_bound_error = []
for numDemo in numDemos:  
    true_perf_ratio = []
    predicted = []
    bound_error = []  
      
    print("=========", numDemo, "=========")
    for rep in range(num_reps):
        filename = "NoisyDemo_numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) +  "_rolloutLength" + str(rolloutLength) +   "_stochastic" + str(stochastic) + "_randActionProb" + str(randActionProb) + "00000_rep" + str(rep)+ ".txt"
        #print filename
        f = open(filePath + filename,'r')   
        f.readline()                                #clear out comment from buffer
        actual = (float(f.readline())) #get the true ratio 
        #print("actual", actual)
        f.readline()                                #clear out ---
        wfcb = (float(f.readline())) #get the worst-case feature count bound
        f.readline()  #clear out ---
        samples = []
        for line in f:                              #read in the mcmc chain
            val = float(line)                       
            samples.append(float(line))
        f.close()
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
            
        
        
        #print("upper bound", upper_bnd)
        predicted.append(upper_bnd)
        true_perf_ratio.append(actual)
        bound_error.append(upper_bnd - actual)
    accuracy = 0.0
    for i in range(len(predicted)):
        if (predicted[i] >= true_perf_ratio[i]) or np.abs(predicted[i] - true_perf_ratio[i]) < tol:
            accuracy += 1.0
    accuracy = accuracy / len(predicted)
    #print(bound_type)
    #print(predicted)
    #print("accuracy", accuracy)
    accuracies.append(accuracy)
    average_bound_error.append(np.mean(bound_error))
    #print()
print(average_bound_error)
c = next(color)
plt.figure(1)
rects1 = plt.bar(index + cnt * bar_width,average_bound_error, bar_width,
             alpha=opacity, label="WFCB", color=c)
plt.figure(2)
rects2 = plt.bar(index + cnt * bar_width,accuracies, bar_width,
             alpha=opacity, label="WFCB", color=c)

cnt += 1





for alpha in alphas:
    print("******", alpha)
    bound_type = bounds[0]
    accuracies = []
    average_bound_error = []
    for numDemo in numDemos:  
        true_perf_ratio = []
        predicted = []
        bound_error = []  
          
        print("=========", numDemo, "=========")
        for rep in range(num_reps):
            #print("rep",rep)
            filename = "NoisyDemo_numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) +  "_rolloutLength" + str(rolloutLength) +   "_stochastic" + str(stochastic) + "_randActionProb" + str(randActionProb) + "00000_rep" + str(rep)+ ".txt"
            #print filename
            f = open(filePath + filename,'r')   
            f.readline()                                #clear out comment from buffer
            actual = (float(f.readline())) #get the true ratio 
            #print("actual", actual)
            f.readline()                                #clear out ---
            wfcb = (float(f.readline())) #get the worst-case feature count bound
            f.readline()  #clear out ---
            samples = []
            for line in f:                              #read in the mcmc chain
                val = float(line)                       
                samples.append(float(line))
            f.close()
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
                
            
            
            #print ("upper bound", upper_bnd)
            predicted.append(upper_bnd)
            true_perf_ratio.append(actual)
            bound_error.append(upper_bnd - actual)
        accuracy = 0.0
        for i in range(len(predicted)):
            if (predicted[i] >= true_perf_ratio[i]) or np.abs(predicted[i] - true_perf_ratio[i]) < tol:
                accuracy += 1.0
        accuracy = accuracy / len(predicted)
        #print(bound_type)
        #print(predicted)
        #print("accuracy", accuracy)
        accuracies.append(accuracy)
        average_bound_error.append(np.mean(bound_error))
        #print()
    print(average_bound_error)
    c = next(color)
    plt.figure(1)
    rects1 = plt.bar(index + cnt * bar_width,average_bound_error, bar_width,
                 alpha=opacity, label="c = "+str(alpha), color=c)

    plt.figure(2)
    rects2 = plt.bar(index + cnt * bar_width,accuracies, bar_width,
                 alpha=opacity, label="c = "+str(alpha), color=c)


    cnt += 1
plt.figure(1)
plt.axis([0,3,-0.2, 3])
plt.yticks(fontsize=18)
plt.xticks(index + 2.5*bar_width, ('1', '5', '9'), fontsize=18)
plt.legend(fontsize=18)
plt.xlabel('number of demonstrations',fontsize=19)
plt.ylabel('average bound error',fontsize=19)
plt.tight_layout()
plt.savefig("./figs/noisydemo_bound_error_overAlpha.png") 

plt.figure(2)
plt.yticks(fontsize=18)
plt.xticks(index + 2.5*bar_width, ('1', '5', '9'), fontsize=18)
plt.legend(fontsize=18, loc='lower left')
plt.xlabel('number of demonstrations',fontsize=19)
plt.ylabel('accuracy',fontsize=19)
plt.tight_layout()
plt.savefig("./figs/noisydemo_accuracy_overAlpha.png") 
plt.show()

