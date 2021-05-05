import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import seaborn as sn

import scipy.integrate as integrate

import time
import progressbar

def total_events(s):
    return 0.5*((s)/(s+1))*129.9e6

def spatial_distribution(theta):
    return (1)/(0.5*np.pi)*(1-(np.cos(theta)**2))

def spatial_partion(viewport_distance,slit_height,slit_width):
    beta = np.arctan(0.5*slit_height/viewport_distance)        
    alpha = np.arctan(0.5*slit_width/viewport_distance)
    
    full_sphere = 4*np.pi*viewport_distance**2
    
    slit_sphere = alpha*(viewport_distance**2)*2*np.sin(beta)
    
    dipol_dist = (integrate.quad(spatial_distribution,0.5*np.pi-beta,0.5*np.pi+beta)[0])/(2*beta/np.pi)#
    
    slit_sphere = slit_sphere * dipol_dist
        
    return slit_sphere/full_sphere
    
if __name__ == "__main__":
    angle = np.arctan(1.7/5.4)
    print "integration of dist over slit"    
    print integrate.quad(spatial_distribution,0.5*np.pi-angle,0.5*np.pi+angle)[0]
    print "classical partial"    
    print 2*angle/np.pi
    print "dipole improvement"    
    print (integrate.quad(spatial_distribution,0.5*np.pi-angle,0.5*np.pi+angle)[0])/(2*angle/np.pi)
    print "events per second"
    print spatial_partion(5.4,3.4,0.2)*total_events(1.2)