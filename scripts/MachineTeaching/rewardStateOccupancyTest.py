import numpy as np

gamma = 0.5
T_pi = np.array([[0,0,0,0,0,0],
        [1,0,0,0,0,0],
        [0,0,0,0,0,1],
        [1,0,0,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0]])
 
print np.linalg.inv(np.eye(6) - gamma * T_pi)
