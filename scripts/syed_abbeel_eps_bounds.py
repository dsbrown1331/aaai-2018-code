# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:13:38 2017

@author: daniel
"""
import numpy as np
gamma = 0.95
delta = 0.95
k = 8
eps1 = 10
m = 0
while(eps1 > 0.93725):
    m = m + 1
    print "count", m
    eps1 = 3.0/(1.0-gamma) * np.sqrt(2.0/m * np.log(2.0*k/delta))
    eps2 = 1.0 / (1.0-gamma) * np.sqrt(2.0 * k / m * np.log( 2.0 * k / delta))
    print 'syed', eps1
    print 'abbeel', eps2
