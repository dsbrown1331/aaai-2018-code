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
num_reps = 20
rolloutLength = 100
numDemos = [1,2,3]
tol = 0.0001
gamma = 0.95
burn = 100 #TODO play with this
skip = 20
delta_conf = 0.95
bounds = ["WFCB", "VaR 99", "VaR 95", "VaR 90"]
epsilons = [0.01, 0.05, 0.1, 0.2, 0.5,1.0]
fmts = ['o-','s--','^-.', '*:','>-','d--']
b = 2 * 1.0/(1.0 - gamma)  #upper bound on random variables for concentration inequalities
c = 10 #truncation value for MPeBCollapsed (Phil's method)

filePath = "../../data/enoughIsEnough/experiment_toy/"


print "UPPER BOUND ON SAMPLES IS b =", b

#for each bound type
for bound_type in bounds:
    print bound_type
    percent_converged_in_max_demos = []
    ave_demos_to_converge = []  
    accuracies = []  
    ave_bound_errors = []
    for eps in epsilons:
        print eps
        #for each repetition
        num_to_get_enough = []
        predicted_bounds = []
        actual_errors = []
        bound_errors = []
        for rep in range(num_reps):
            #extend the number of demos until predicted bound is below epsilon
            enoughDemo = np.nan
            predicted = np.nan
            ground_truth = np.nan
            for numDemo in numDemos:
                ###Get VaR confidence bound
                filename = filePath + "numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) +  "_rolloutLength" + str(rolloutLength) + "_stochastic0_rep" + str(rep)+ ".txt"
                #print filename
                f = open(filename,'r')   
                f.readline()                                #clear out comment from buffer
                true_return = (float(f.readline())) #get the true opt policy exp return
                f.readline()
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
                elif bound_type == "VaR 99":
                    upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.99, delta_conf)
                elif bound_type == "VaR 95":
                    upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.95, delta_conf)
                elif bound_type == "VaR 90":
                    upper_bnd = bound_methods.percentile_confidence_upper_bnd(burned_samples, 0.9, delta_conf)
                elif bound_type == "aveVal":
                    upper_bnd = np.mean(burned_samples)
                elif bound_type == "AM":
                    upper_bnd = -1*bound_methods.anderson_lower_bnd(-np.array(burned_samples), delta_conf)
                elif bound_type == "MPeB":
                    upper_bnd = bound_methods.empirical_bernstein_bnd(burned_samples, delta_conf, b)
                elif bound_type == "WFCB":
                    upper_bnd = wfcb

                ###check if VaR conf bound is below eps
                if upper_bnd < eps:
                    enoughDemo = numDemo
                    predicted = upper_bnd
                    ground_truth = actual
                    break
#            if enoughDemo is np.nan:
#                print "NEVER CONVERGED"
#            else:
#                print enoughDemo
            #print "enough", enoughDemo
            #print "predicted", predicted   
            #print "truth", ground_truth
            predicted_bounds.append(predicted)
            actual_errors.append(ground_truth)
            num_to_get_enough.append(enoughDemo)
            bound_errors.append(predicted - ground_truth)
        percent_converged_in_max_demos.append((num_reps - np.sum(np.isnan(num_to_get_enough)))/float(num_reps))
        ave_demos_to_converge.append(np.nanmean(num_to_get_enough))
        ave_bound_errors.append(np.nanmean(bound_errors))
        accuracy = 0.0
        count = 0.0
        #print "predicted bounds", predicted_bounds
        for i in range(len(predicted_bounds)):
            if predicted_bounds[i] is not np.nan:
                #print "pred", predicted_bounds[i]
                count += 1.0
                if (predicted_bounds[i] >= actual_errors[i]) or np.abs(predicted_bounds[i] - actual_errors[i]) < tol:
                    accuracy += 1.0
        #print "COUNT", count
        if count == 0.0:
            accuracy = np.nan
        else:
            accuracy = accuracy / count
        accuracies.append(accuracy)
    print "accuracies", accuracies
    print "percent converged", percent_converged_in_max_demos
    print "ave demos to converge", ave_demos_to_converge
    print "ave bound errors", ave_bound_errors
    fig_cnt = 1
    plt.figure(fig_cnt)
    #plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
    plt.xlabel("epsilon safety margin", fontsize=19)
    plt.ylabel("% converged within " + str(numDemos[-1]) +" demos", fontsize=19)
    #plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
    plt.plot(epsilons, percent_converged_in_max_demos, fmts[bounds.index(bound_type)], label=bound_type, lw=2)
    #plot dotted line across diagonal
    #plt.plot([0,true_perf_ratio[-1]],[0,true_perf_ratio[-1]], 'k:')
    plt.xticks(fontsize=18) 
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.01],[0.0,0.2,0.4,0.6,0.8,1.0,''], fontsize=18) 
    plt.legend(loc='best', fontsize=19)
    plt.tight_layout()            
    plt.savefig("noTerminalNoisyTransition_percentConverged.png")

    #plot the average number to converge if converged
    fig_cnt = 2
    plt.figure(fig_cnt)
    #plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
    plt.xlabel("epsilon safety margin", fontsize=19)
    plt.ylabel("num demos to reach epsilon error", fontsize=19)
    #plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
    plt.plot(epsilons, ave_demos_to_converge, fmts[bounds.index(bound_type)], label=bound_type, lw=2)
    #plot dotted line across diagonal
    #plt.plot([0,true_perf_ratio[-1]],[0,true_perf_ratio[-1]], 'k:')
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18) 
    plt.legend(loc='best', fontsize=19)
    plt.tight_layout()            
    plt.savefig("noTerminalNoisyTransition_aveDemosToConverge.png")


    #plot the average number to converge if converged
    fig_cnt = 3
    plt.figure(fig_cnt)
    #plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
    plt.xlabel("epsilon safety margin", fontsize=19)
    plt.ylabel("accuracy", fontsize=19)
    #plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
    plt.plot(epsilons, accuracies, fmts[bounds.index(bound_type)], label=bound_type, lw=2)
    #plot dotted line across diagonal
    #plt.plot([0,true_perf_ratio[-1]],[0,true_perf_ratio[-1]], 'k:')
    plt.xticks(fontsize=18) 
    plt.yticks([0.90,0.92, 0.94, 0.96, 0.98, 1.0, 1.001],[0.90, 0.92, 0.94, 0.96, 0.98, 1.0,''], fontsize=18)
    plt.legend(loc='best', fontsize=19)
    plt.tight_layout()            
    plt.savefig("noTerminalNoisyTransition_aveAccuracies.png")

    #plot the average number to converge if converged
    fig_cnt = 4
    plt.figure(fig_cnt)
    #plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
    plt.xlabel("epsilon safety margin", fontsize=19)
    plt.ylabel("ave bound error", fontsize=19)
    #plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
    plt.plot(epsilons, ave_bound_errors, fmts[bounds.index(bound_type)], label=bound_type, lw=2)
    #plot dotted line across diagonal
    #plt.plot([0, epsilons[-1]],[0,epsilons[-1]], 'k:')
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18) 
    plt.legend(loc='best', fontsize=19)
    plt.tight_layout()            
    plt.savefig("noTerminalNoisyTransition_aveBoundErrors.png")




#######
#Plot the distribution (histogram) over exp return for opt policy
#####
#just set num demos to anything since it shouldn't matter
numDemo = 1
opt_returns = []
for rep in range(num_reps):

    ###Get VaR confidence bound
    filename = filePath + "numdemos" +  str(numDemo) + "_alpha" + str(alpha) + "_chain" + str(chain_length) +  "_step" + str(step) + "0000_L1sampleflag" + str(sample_flag) +  "_rolloutLength" + str(rolloutLength) + "_stochastic0_rep" + str(rep)+ ".txt"
    #print filename
    f = open(filename,'r')   
    f.readline()                                #clear out comment from buffer
    true_return = (float(f.readline())) #get the true opt policy exp return
    opt_returns.append(true_return)

#plot the average number to converge if converged
fig_cnt = 5
plt.figure(fig_cnt)
#plt.title(r"5x5 navigation domain $\alpha = $" + str(alpha) + ", $step = $" + str(step))
plt.xlabel("expected return of optimal policy", fontsize=19)
plt.ylabel("frequency", fontsize=19)
#plt.errorbar(true_perf_ratio, average_bounds, yerr=stdev_bounds, fmt=fmts[bounds.index(bound_type)], label=bound_type, lw=1)
plt.hist(opt_returns)
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18) 
#plt.legend(loc='best', fontsize=19)
plt.tight_layout()            
plt.savefig("optimalReturnsHistogram.png")






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
