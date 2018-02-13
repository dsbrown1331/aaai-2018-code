import numpy as np
import matplotlib.pyplot as plt
from numpy import nan, inf

filePath = "/home/dsbrown/Code/PeARL_BIRL/data/active_bench/"
size = 8
trajLength = 1
numRuns = 100
maxQueries = 51
startQueries = 5
plotQueries = 30
x = np.zeros((numRuns, maxQueries))
evd_loss = np.full_like(x, np.nan, dtype=np.double)
zero_one_loss = np.full_like(x, np.nan, dtype=np.double)
#evd_loss = np.zeros((numRuns, maxQueries))
#zero_one_loss = np.zeros((numRuns, maxQueries))
fmts = ['-','-.','--', '*:','>-','d--']

count = 0
#optimal teaching errors
for seed in range(1,numRuns+1):
    filename = "opt_" + str(size) + "_trajLength=" + str(trajLength) + "_seed" + str(seed) + ".txt";
    print(filename)
    #TODO: plot policy losses.
    f = open(filePath + filename,'r')

   
    cnt = 0
    for line in f:
        parsed = line.strip().split(",")
        print(parsed)
        evd = float(parsed[1])
        zero_one = float(parsed[2])
#        if(evd == 0 or zero_one == 0):
#            break
        evd_loss[seed-1,cnt] = evd
        zero_one_loss[seed-1,cnt] = zero_one
        cnt += 1
print (evd_loss)
print(zero_one_loss)
print(np.nanmean(evd_loss, axis=0))
plt.figure(1)
plt.plot(range(startQueries,plotQueries),np.nanmean(evd_loss[:,startQueries:plotQueries], axis=0), fmts[count], label='set-cover',linewidth=3.0)
plt.figure(2)
plt.plot(range(startQueries,plotQueries),np.nanmean(zero_one_loss[:,startQueries:plotQueries],axis=0), fmts[count], label='set-cover',linewidth=3.0)

count += 1
#entropy errors
for seed in range(1,numRuns+1):
    filename = "entropy_" + str(size) + "_trajLength=" + str(trajLength) + "_seed" + str(seed) + ".txt";
    print(filename)
    #TODO: plot policy losses.
    f = open(filePath + filename,'r')

   
    cnt = 0
    for line in f:
        parsed = line.strip().split(",")
        print(parsed)
        evd = float(parsed[1])
        zero_one = float(parsed[2])
#        if(evd == 0 or zero_one == 0):
#            break
        evd_loss[seed-1,cnt] = evd
        zero_one_loss[seed-1,cnt] = zero_one
        cnt += 1
print (evd_loss)
print(zero_one_loss)
print(np.nanmean(evd_loss, axis=0))
plt.figure(1)
plt.plot(range(startQueries,plotQueries),np.nanmean(evd_loss[:,startQueries:plotQueries], axis=0), fmts[count], label='entropy',linewidth=3.0)
plt.figure(2)
plt.plot(range(startQueries,plotQueries),np.nanmean(zero_one_loss[:,startQueries:plotQueries],axis=0),fmts[count], label='entropy',linewidth=3.0)

count += 1
#VaR errors
for seed in range(1,numRuns+1):
    filename = "var_" + str(size) + "_trajLength=" + str(trajLength) + "_seed" + str(seed) + ".txt";
    print(filename)
    #TODO: plot policy losses.
    f = open(filePath + filename,'r')

   
    cnt = 0
    for line in f:
        parsed = line.strip().split(",")
        print(parsed)
        evd = float(parsed[1])
        zero_one = float(parsed[2])
#        if(evd == 0 or zero_one == 0):
#            break
        evd_loss[seed-1,cnt] = evd
        zero_one_loss[seed-1,cnt] = zero_one
        cnt += 1
print (evd_loss)
print(zero_one_loss)
print(np.nanmean(evd_loss, axis=0))
plt.figure(1)
plt.plot(range(startQueries,plotQueries),np.nanmean(evd_loss[:,startQueries:plotQueries], axis=0), fmts[count], label='VaR',linewidth=3.0)
plt.figure(2)
plt.plot(range(startQueries,plotQueries),np.nanmean(zero_one_loss[:,startQueries:plotQueries],axis=0),fmts[count], label='VaR',linewidth=3.0)

count += 1
#random errors
for seed in range(1,numRuns+1):
    filename = "random_" + str(size) + "_trajLength=" + str(trajLength) + "_seed" + str(seed) + ".txt";
    print(filename)
    #TODO: plot policy losses.
    f = open(filePath + filename,'r')

   
    cnt = 0
    for line in f:
        parsed = line.strip().split(",")
        print(parsed)
        evd = float(parsed[1])
        zero_one = float(parsed[2])
#        if(evd == 0 or zero_one == 0):
#            break
        evd_loss[seed-1,cnt] = evd
        zero_one_loss[seed-1,cnt] = zero_one
        cnt += 1
print (evd_loss)
print(zero_one_loss)
print(np.nanmean(evd_loss, axis=0))
plt.figure(1)
plt.plot(range(startQueries,plotQueries),np.nanmean(evd_loss[:,startQueries:plotQueries], axis=0), fmts[count], label='random',linewidth=3.0)
plt.figure(2)
plt.plot(range(startQueries,plotQueries),np.nanmean(zero_one_loss[:,startQueries:plotQueries],axis=0),fmts[count], label='random',linewidth=3.0)

plt.figure(1)
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18) 
plt.xlabel("Queries", fontsize=19)
plt.ylabel("EVD policy Loss", fontsize=19)
plt.legend(loc='best', fontsize=19)
plt.tight_layout()

plt.figure(2)
plt.xticks(fontsize=18) 
plt.yticks(fontsize=18) 
plt.xlabel("Queries", fontsize=19)
plt.ylabel("0-1 policy Loss", fontsize=19)
plt.legend(loc='best', fontsize=19)
plt.tight_layout()
plt.show()

