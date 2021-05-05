# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 17:22:53 2017

@author: Stefan
"""
from __future__ import print_function
#Loading of packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

#starting parameters
positions = [[0.00001,-0.00001]]
stepsize = 3*10**-6
residuum = 1*10**-9

Epsilon = 8.854187817*10**-12
Q = 1.6021766208*10**-19
M = 6.642*10**-26
M_Be = 1.496508187*10**-26


#Auxilliary functions
#x1 and x2 are Ca
def potential(x1, x2, omega):
    return 0.5*M*(omega**2.0)*(x1**2.0 + x2**2.0)+(Q**2.0/(4.0*np.pi*Epsilon))*(1.0/(np.abs(x1-x2)))

#x1 and x2 are Ca    
def potential_derivation(x1, x2, omega):
    return 0.5*M*(omega**2.0)*(2.0*x1 + x2**2.0)-(Q**2.0/(4.0*np.pi*Epsilon))*(1.0/(x1-x2)**2)

#x1 and x2 are Be, x3 is Ca
def three_potential(x1,x2,x3, omega):
    return 0.5*M_Be*(omega**2.0)*(x1**2.0 + x2**2.0)+0.5*M*(omega**2.0)*(x3**2.0)+(Q**2.0/(4.0*np.pi*Epsilon))*((1.0/(np.abs(x1-x2)))+(1.0/(np.abs(x1-x3)))+(1.0/(np.abs(x2-x3))))

#x1 and x2 are Ca, x3 is Be
def three_potential_two(x1,x2,x3, omega):
    return 0.5*M*(omega**2.0)*(x1**2.0 + x2**2.0)+0.5*M_Be*(omega**2.0)*(x3**2.0)+(Q**2.0/(4.0*np.pi*Epsilon))*((1.0/(np.abs(x1-x2)))+(1.0/(np.abs(x1-x3)))+(1.0/(np.abs(x2-x3))))

#x1 and x2 are Be, x3 and x4 are Ca
def four_potential(x1,x2,x3,x4, omega):
    return 0.5*M_Be*(omega**2.0)*(x1**2.0 + x2**2.0)+0.5*M*(omega**2.0)*(x3**2.0 + x4**2)+(Q**2.0/(4.0*np.pi*Epsilon))*((1.0/(np.abs(x1-x2)))+(1.0/(np.abs(x1-x3)))+(1.0/(np.abs(x2-x3)))+(1.0/(np.abs(x1-x4)))+(1.0/(np.abs(x2-x4)))+(1.0/(np.abs(x3-x4))))

#main runtime

#Runtime for two Calcium Ions
def two_ca(omega):
    print("relaxation of two Ca ions")
    print("trap-frequency: " + str(omega/(np.pi*2)) + "Hz")
    for i in range(10000):
        random = np.random.rand()
        if potential_derivation(positions[-1][0], positions[-1][1], omega) <= 0:
            new_x1 = positions[-1][0] + random*stepsize
        else:
            new_x1 = positions[-1][0] - random*stepsize
        if potential(new_x1,positions[-1][1], omega) < potential(positions[-1][0],positions[-1][1], omega):
            positions.append([new_x1, -new_x1])
            if (np.abs(positions[-1][0]-positions[-2][0])) < residuum:
                print("needed: " + str(i) + " Iterations!")
                print("accepted steps: " + str(len(positions)))
                print("fraction: " + str(float(len(positions))/float(i)))
                break

    plt.clf()
    plt.plot(range(len(positions)),np.array(positions).T[0], color="b", label='calcium 1')
    plt.plot(range(len(positions)),np.array(positions).T[1], color="b", label='calcium 2')
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    text = (r'Calcium: ' + str(positions[-1][0]) + "\n")
    plt.text(5, -0.00001, text, None, bbox=props, fontsize=12)
    plt.ylabel(r'position in $[\mu m]$')
    plt.xlabel('steps')
    plt.legend(loc='best')
    plt.title('Two calcium ions')
    print("Positions of two Ca-Ions:")
    print(positions[-1])
    plt.savefig('img/two_ca.png', dpi=500)
    plt.savefig('img/two_ca.pdf')    
    plt.show()
    return positions[-1]

#Runtime For Be-Ca-Be Crystal
def one_ca_two_be(omega):
    print("positions for Be-Ca-Be")    
    positions = two_ca(omega) 
    positions2 = [[positions[0], positions[1],0]]
    #positions2 = [[0.00005, -0.00005,0]]
    for i in range(50000000):
        random = np.random.rand()
        new_x1 = positions2[-1][0] + (random-0.5)*2*stepsize
        if three_potential(new_x1, -new_x1, 0, omega) < three_potential(positions2[-1][0], positions2[-1][1], 0, omega):
            positions2.append([new_x1, -new_x1, 0])
            if (len(positions2) > 2):
                if (np.abs(positions2[-1][0]-positions2[-2][0])) < residuum:
                    print("needed: " + str(i) + " Iterations!")
                    print("accepted steps: " + str(len(positions2)))
                    print("fraction: " + str(float(len(positions2))/float(i)))
                    break
                
    plt.clf()
    plt.plot(range(len(positions2)),np.zeros(len(positions2)), color="b", label='Berrylium')
    plt.plot(range(len(positions2)),np.array(positions2).T[0], color="r", label='Calcium 1')
    plt.plot(range(len(positions2)),np.array(positions2).T[1], color="r", label='Calcium 2')
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    text = (r'Calcium: ' + '0' + "\n" + 'Beryllium: ' + str(positions2[-1][0]))
    plt.text(3, -0.000003, text, None, bbox=props, fontsize=12)
    plt.ylabel('position')
    plt.xlabel('steps')
    plt.legend(loc='best')
    plt.title('Two Beryllium, one Calcium ion')
    print("Positions of two Be-Ions and one Ca-Ion:")
    print(positions2[-1])
    plt.savefig('img/two_be_one_ca.png', dpi=300)
    plt.show()
    return positions2[-1]
    
#Runtime for Ca-Be-ca Crystal
def one_be_two_ca(omega):
    print("positions for Ca-Be-Ca")    
    positions = two_ca(omega) 
    positions2 = [[positions[0], positions[1],0]]
    #positions2 = [[0.00005, -0.00005,0]]
    for i in range(500000):
        random = np.random.rand()
        new_x1 = positions2[-1][0] + (random-0.5)*2*stepsize
        if three_potential_two(new_x1, -new_x1, 0, omega) < three_potential_two(positions2[-1][0], positions2[-1][1], 0, omega):
            positions2.append([new_x1, -new_x1, 0])
            if (len(positions2) > 2):
                if (np.abs(positions2[-1][0]-positions2[-2][0])) < residuum:
                    print("needed: " + str(i) + " Iterations!")
                    print("accepted steps: " + str(len(positions2)))
                    print("fraction: " + str(float(len(positions2))/float(i)))
                    break
                
    plt.clf()
    plt.plot(range(len(positions2)),np.zeros(len(positions2)), color="b", label='Calcium')
    plt.plot(range(len(positions2)),np.array(positions2).T[0], color="r", label='Beryllium 1')
    plt.plot(range(len(positions2)),np.array(positions2).T[1], color="r", label='Beryllium 2')
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    text = (r'Berillium: ' + '0' + "\n" + 'Calcium: ' + str(positions2[-1][0]))
    plt.text(3, -0.000003, text, None, bbox=props, fontsize=12)
    plt.ylabel('position')
    plt.xlabel('steps')
    plt.legend(loc='best')
    plt.title('Two Calcium, one Beryllium ions')
    print("Positions of two Ca-Ions and one Be-Ion:")
    print(positions2[-1])
    plt.savefig('img/two_ca_one_be.png', dpi=300)
    plt.show()
    return positions2[-1]

#Runtime for Be-Ca-Ca-Be Crystal
def two_ca_two_be(omega):
    positions = two_ca(omega)
    positions3 = [[2*positions[0], 2*positions[1], positions[0], positions[1]]]
    print("Be-Ca-Ca-Be")    
    #positions3 = [[0.0001,0.00005, -0.00005,-0.0001]]
    for i in range(5000000):
        random = np.random.rand()
        new_x1 = positions3[-1][2] + (random-0.5)*stepsize
        if four_potential(positions3[-1][0], positions3[-1][1], new_x1, -new_x1, omega) < four_potential(positions3[-1][0], positions3[-1][1], positions3[-1][2], positions3[-1][3], omega):
            positions3.append([positions3[-1][0], positions3[-1][1], new_x1, -new_x1])
            random = np.random.rand()
            new_x1 = positions3[-1][0] + (random-0.5)*stepsize
        if four_potential(new_x1, -new_x1, positions3[-1][2],positions3[-1][3], omega) < four_potential(positions3[-1][0], positions3[-1][1], positions3[-1][2], positions3[-1][3], omega):
                positions3.append([new_x1, -new_x1, positions3[-1][2], positions3[-1][3]])
        if (len(positions3) > 2):
            if ((np.abs(positions3[-1][0]-positions3[-2][0])) < 0.5*residuum) & ((np.abs(positions3[-1][2]-positions3[-2][2])) < 0.5*residuum):
                print("needed: " + str(i) + " Iterations!")
                print("accepted steps: " + str(len(positions3)))
                print("fraction: " + str(float(len(positions3))/float(i)))
                break
    
    print("accepted steps: " + str(len(positions3)))        
    plt.clf()
    plt.plot(range(len(positions3)),np.array(positions3).T[2], color="b", label='Calcium 1')
    plt.plot(range(len(positions3)),np.array(positions3).T[3], color="b", label='Calcium 2')
    plt.plot(range(len(positions3)),np.array(positions3).T[0], color="r", label='Beryllium 1')
    plt.plot(range(len(positions3)),np.array(positions3).T[1], color="r", label='Beryllium 2')
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    text = (r'Calcium: ' + str(positions3[-1][2]) + "\n" + 'Beryllium: ' + str(positions3[-1][0]))
    plt.text(0.5, positions3[-1][1]- 0.000003, text, None, bbox=props, fontsize=12)
    plt.ylabel('position')
    plt.xlabel('steps')
    plt.legend(loc='best')
    plt.title('Two Beryllium, Two Calcium ions')
    print("Positions of two Ca-Ions and Two Beryllium Ions:")
    print(positions3[-1])
    plt.savefig('img/two_ne_two_ca.png', dpi=300)
    plt.show()