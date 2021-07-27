from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

#Definition of stuff
hbar = 1.054571800e-34
M = 6.642*10**-26
M_Be = 1.496508187*10**-26


#Auxilliary Functions
def DW_part(phonons, pulse_vec, mode_freq):
    return (((hbar*(pulse_vec)**2)/(M*mode_freq))*(float(phonons)+0.5))

#Main Debye-Waller Function
def DW_two_ca(Phonons_breathing, Phonons_rocking_x, Phonons_rocking_y, Frequency_Breathing, Frequency_Rocking_x, Frequency_Rocking_y, vec_z, vec_x, vec_y):
    print (((2*np.pi)/(397e-9))**2)
    print (((2*np.pi)/(854e-9))**2)
    print (((2*np.pi)/(729e-9))**2)

    print (-(DW_part(Phonons_rocking_x, vec_x, Frequency_Rocking_x))-(DW_part(Phonons_rocking_y, vec_y, Frequency_Rocking_y))-(DW_part(Phonons_breathing, vec_z, Frequency_Breathing)))
    return np.exp((((2*np.pi)/(397e-9))**2)*(-(DW_part(Phonons_rocking_x, vec_x, Frequency_Rocking_x))-(DW_part(Phonons_rocking_y, vec_y, Frequency_Rocking_y))-(DW_part(Phonons_breathing, vec_z, Frequency_Breathing))))

#Now calculate everything from experiment parameters
def DW_two_complete(incoming_angle, outgoing_angle, trap_z, trap_x, trap_y, phonon_rocking_x, phonon_rocking_y, phonon_breathing):

    breathing_freq = np.sqrt(3)*(trap_z)
    rocking_x = np.sqrt(trap_x**2 - trap_z**2)
    rocking_y = np.sqrt(trap_y**2 - trap_z**2)
    
        
    q_z = (np.cos(incoming_angle + np.pi) - np.cos(outgoing_angle))
    #print q_z
    q_y = (np.sin(incoming_angle + np.pi) - np.sin(outgoing_angle))
    #print q_y
    q_x = 0


    #q_x = ((((2*np.pi)/(729e-9))**2)*np.cos(18))-(((2*np.pi)/(854e-9))**2)*np.cos(18))
    #q_y = ((((2*np.pi)/(729e-9))**2)*-np.sin(18))+(((2*np.pi)/(854e-9))**2)*np.sin(18))-((((2*np.pi)/(397e-9))**2)*np.sin(18))

    return DW_two_ca(phonon_breathing, phonon_rocking_x, phonon_rocking_y, breathing_freq, rocking_x, rocking_y, q_z, q_x, q_y)

if __name__ == "__main__":
    print('testing debye waller module')
    omega_z = 2.0*np.pi*750*10**3 #trap frequency along z-axis
    omega_x = 2.0*np.pi*2430*10**3 #trap frequency along x-axis
    omega_y = 2.0*np.pi*1570*10**3 #trap frequency along y-axis

    Phonons_breath = 7
    Phonons_rocking_x = 12
    Phonons_rocking_y = 12
    
    xes = np.linspace(0, 1000, num=5001)
    yes = []    
    
    for item in xes:
        yes.append(DW_two_complete(0.25*np.pi, -0.5*np.pi, omega_z, omega_x, omega_y, item, item, item))
    
    plt.xlabel('phonons')
    plt.ylabel('dwf')
    plt.plot(xes, yes)
    plt.savefig('dwf_raw', dpi=300)
    
    plt.xlim([0, 100])
    plt.savefig('dwf_semi', dpi=300)
    
    plt.xlim([0, 25])
    plt.savefig('dwf_fine', dpi=300)