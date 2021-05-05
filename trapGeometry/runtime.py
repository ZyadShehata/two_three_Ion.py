import numpy as np

import harmonic_distribution as hd
import Contrast_losses as cont
import spatial as sp
import crystallization as crys
import debye_waller as dw

import matplotlib.pyplot as plt
import seaborn as sn

from PIL import Image

#GLOBAL CONFIG
#EXPERIMENTAL SETUP
saturation_param = 0.6
bin_number = 5 #as integer, describes in how many bins the coherence time is split
slit_width = 2.0 #in mm
slit_height = 4.5 #in cm
viewport_distance = 4.8 #in cm
quantum_efficiency = 0.99*0.25*0.5*0.75 #0.99 objective, 0.5 couple in, 0.5 quantum path beam splitter, 0.75 quantum efficiency SPAPD (Count Blue)
beam_angle = 0.25*np.pi
center_angle = -0.5*np.pi #horizontal right out of trap = 0 rad --> to turn right, use negative values

omega_z = 2.0*np.pi*750*10**3 #trap frequency along z-axis
omega_x = 2.0*np.pi*2430*10**3 #trap frequency along x-axis
omega_y = 2.0*np.pi*1570*10**3 #trap frequency along y-axis

Phonons_breath = 7
Phonons_rocking_x = 12
Phonons_rocking_y = 12

#Ion Stuff
#Calculate Ion Positions
#For Two Cas
ion_distance = crys.two_ca(omega_z)[0] #distance between ions, calculated from trap frequency
#For Ca-Be-Ca Crystal
#ion_distance = crys.one_be_two_ca(omega_z)[0]

coherence_time = 7.7e-9 #in seconds

#Debye-Waller Factor
debye_waller = dw.DW_two_complete(beam_angle, center_angle, omega_z, omega_x, omega_y, Phonons_rocking_x, Phonons_rocking_y, Phonons_breath)


#Statistical Stuff
significance = 5 #in sigmas for poissonian threshhold
threshhold = 100 # arbitrary threshhold, counts ions in bins

#Script doing stuff
print "Total Quantum Efficiency"
print str(quantum_efficiency)
print "Estimated Debye-Waller-Factor"
print str(debye_waller)
print "Expected g^2 T=0:"
exp_gtwo = cont.total_contrast(bin_number,saturation_param,center_angle, beam_angle,slit_width,ion_distance,debye_waller)
print exp_gtwo
print "Visibility of g^2"
print str((2*saturation_param*debye_waller)/(saturation_param**2 + debye_waller**2))
print "Events per Second in whole spatial regime"
print sp.total_events(saturation_param)
print "Events per second on slit"
print sp.spatial_partion(viewport_distance,slit_height,float(slit_width)/10)*sp.total_events(saturation_param)
print "fraction of events in promille"
print (sp.spatial_partion(viewport_distance,slit_height,float(slit_width)/10)*sp.total_events(saturation_param))/sp.total_events(saturation_param)*1000
print "measured counts per second with losses"
per_second_losses = sp.spatial_partion(viewport_distance,slit_height,float(slit_width)/10)*sp.total_events(saturation_param)*quantum_efficiency
print per_second_losses
#print "number of bins between two photon events"
bins_per_interphotontime = (1/per_second_losses)/((coherence_time)/(bin_number))
#print bins_per_interphotontime
print "measurement time for arbitrary threshhold in hours (Threshhold: "+ str(threshhold) +")"
print (1./3600)*threshhold*((coherence_time**2)*(1/((coherence_time/bin_number)*(cont.poissonian(coherence_time*per_second_losses,2)))))
print "measurementtime limted by poissonian noise in hours (Threshhold: "+ str((((exp_gtwo)*significance**2)/((exp_gtwo-1)**2))) +", significance: "+ str(significance) + " sigma)"
print (1./3600)*(((exp_gtwo)*significance**2)/((exp_gtwo-1)**2))*((coherence_time**2)*(1/((coherence_time/bin_number)*(cont.poissonian(coherence_time*per_second_losses,2)))))
#visualization
img = Image.open('trap.png')
plt.grid(False)
plt.axis('off')
plt.imshow(img)
x = np.cos(center_angle)*512.+512
y = -np.sin(center_angle)*512.+512
x2 = np.cos(beam_angle)*512.+512
y2 = -np.sin(beam_angle)*512.+512
plt.plot([512,x],[512,y], color="blue", label="outgoing beam")
plt.plot([512,x2],[512,y2], color="red", label="incoming beam")
plt.legend(loc='best')
plt.savefig('img/scheme.png', dpi=500)
plt.savefig('img/scheme.pdf')
plt.show()
plt.clf()
cont.visualize_gtwo(center_angle, beam_angle,slit_width,saturation_param,ion_distance,debye_waller)