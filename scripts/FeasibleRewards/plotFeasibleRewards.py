import numpy as np
import matplotlib.pyplot as plt

f = open("../../data/enoughIsEnough/experiment_toy_feasible/demo_from_state11_L1.txt")
count = 0
pts = []
min_so_far = 0
for line in f:
    count += 1
    reward = line.strip().split(",")
    pts.append(reward)
    if float(reward[0]) < min_so_far:
        min_so_far = float(reward[0])

feasible = np.asarray(pts)
print feasible   
#plt.plot(feasible[:,0], feasible[:,1],'k.')
#plt.axis([-1,0,-1,0])
#plt.show()

print min_so_far


    
