# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:47:30 2017

@author: Stefan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sn

def harmonic_position(A,omega,t):
    return A*np.sin(omega*t*2)
    
def position_distribution(x_m, x, omega):
    return 1/(omega*(x_m**2-x**2)**0.5)

if __name__ == "__main__":
    #Runtime for two Calcium Ions
    data = []    
    omega = 1*np.pi
    for i in range(500000):
        random = np.random.rand()
        data.append(harmonic_position(1,omega,random))
    bins = plt.hist(data, 200, normed=1, facecolor='green', alpha=0.75)
    plt.plot(bins[1], position_distribution(1, bins[1], omega), color='red')
    plt.ylim([0,3])