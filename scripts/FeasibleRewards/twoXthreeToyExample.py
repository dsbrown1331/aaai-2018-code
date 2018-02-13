# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:46:10 2017

@author: daniel
 """

import numpy as np

nStates = 6
nFeatures = 2
gamma = 0.9

T_pi = np.array([[0,0,0,0,0,0],
                 [1,0,0,0,0,0],
                 [0,0,0,0,0,1],
                 [1,0,0,0,0,0],
                 [0,0,0,1,0,0],
                 [0,0,0,0,1,0]])

T_up = np.array([[0,0,0,0,0,0],
                 [0,1,0,0,0,0],
                 [0,0,1,0,0,0],
                 [1,0,0,0,0,0],
                 [0,1,0,0,0,0],
                 [0,0,1,0,0,0]])
                 
T_down = np.array([[0,0,0,0,0,0],
                 [0,0,0,0,1,0],
                 [0,0,0,0,0,1],
                 [0,0,0,1,0,0],
                 [0,0,0,0,1,0],
                 [0,0,0,0,0,1]])
                 
T_left = np.array([[0,0,0,0,0,0],
                 [1,0,0,0,0,0],
                 [0,1,0,0,0,0],
                 [0,0,0,1,0,0],
                 [0,0,0,1,0,0],
                 [0,0,0,0,1,0]])
T_right = np.array([[0,0,0,0,0,0],
                 [0,0,1,0,0,0],
                 [0,0,1,0,0,0],
                 [0,0,0,0,1,0],
                 [0,0,0,0,0,1],
                 [0,0,0,0,0,1]])

Phi = np.array([[1,0],
                [0,1],
                [1,0],
                [1,0],
                [1,0],
                [1,0]])
#print T_pi
#print T_up
#print T_down
#print T_left
#print T_right
c_up = (np.dot(T_pi - T_up, np.linalg.inv(np.eye(nStates) - gamma * T_pi)))
c_down = (np.dot(T_pi - T_down, np.linalg.inv(np.eye(nStates) - gamma * T_pi)))
c_left = (np.dot(T_pi - T_left, np.linalg.inv(np.eye(nStates) - gamma * T_pi)))
c_right = (np.dot(T_pi - T_right, np.linalg.inv(np.eye(nStates) - gamma * T_pi)))
constraints = np.vstack((c_up,c_down,c_left,c_right))
w_constraints = np.dot(constraints, Phi)
print w_constraints

#remove duplicates and zero rows
to_remove = set()
for i in range(len(w_constraints)):
    for j in range(i+1,len(w_constraints)):
        if np.allclose(w_constraints[i], w_constraints[j]):
            to_remove.add(i)
        if np.allclose(w_constraints[i], np.zeros((1,nFeatures))):
            to_remove.add(i)
            


print to_remove
non_dup_constraints = np.delete(w_constraints,list(to_remove), axis=0)
print non_dup_constraints
