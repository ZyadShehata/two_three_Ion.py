import numpy as np
import matplotlib.pyplot as plt
#Phase parameters

#to calculate the phase factor imprinted on the ions
def delta_of_phi(phi,d_ion):
    return (2*np.pi/397e-9) * (-1*np.cos(np.pi*18/180) - np.cos(np.pi*phi/180)) * d_ion

def delta_of_phi_recoil_free(phi, d_ion):
    pass

def debye_waller_factor(n_breathing=6, n_rocking_1=10, n_rocking_2=10, q_breathing=np.pi*2*600e3, q_rocking_1=np.pi*2*300e3, q_rocking_2=np.pi*2*300e3):
    return 1.

def g_two_nominator(s_star, phi_1, phi_2, d_ion):
    phase = (delta_of_phi(phi_1, d_ion)-delta_of_phi(phi_2, d_ion))/2
    return s_star*np.cos(phase)**2*debye_waller_factor()

def g_two_denominator(s_star, phi_1, phi_2, d_ion):
    return (s_star + np.cos(delta_of_phi(phi_1, d_ion))*debye_waller_factor())*(s_star + np.cos(delta_of_phi(phi_2, d_ion))*debye_waller_factor())

def g_two(s_star, phi_1, phi_2, d_ion):
    return g_two_nominator(s_star, phi_1, phi_2, d_ion)/g_two_denominator(s_star, phi_1, phi_2, d_ion)


if __name__ == "__main__":
    pass
    #do stuff here
    #new trap works with 1- 1.5MHz trap frequency --> 4.3e-6m - 5.6e-6m
    d_ion= 4.3*10**-6
    #according to: 0pi, 0.5*pi, pi, 1.5*pi --> starting phase --> 397 single illumination
    starting_angles = [81., 82., 84., 85.3]
    range_start = 270. - 36.78
    range_stop = 270. + 36.78
    saturation = 1 + 0.7

    #select proper starting phases
    #xes = np.linspace(90.-36.78, 90.+36.78, num=1000)
    #phase is accumulated, chop it with 2pi
    #yes = delta_of_phi(xes, d_ion)%(2*np.pi)
    #plt.figure()
    #plt.plot(xes,yes)
    #plt.show()


    xes = np.linspace(range_start, range_stop, num=1000)
    yes_array = []
    #yes = np.zeros_like(xes)
    #i = 0
    #for x in xes:
    #    yes[i] = g_two(saturation, starting_angle, x, d_ion)
    #    i +=1

    for starting_angle in starting_angles:
        yes_array.append(g_two(saturation, starting_angle, xes, d_ion))

    plt.figure()
    i=0
    for yes in yes_array:
        plt.plot(xes, yes, label=str(starting_angles[i]))
        i += 1
    plt.legend(loc="best")
    plt.show()

