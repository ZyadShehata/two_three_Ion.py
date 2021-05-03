# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:47:30 2017

@author: Stefan
"""

import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import seaborn as sn

sn.set(font_scale=1.70)

import scipy.integrate as integrate

import time
import progressbar

#3d kacke
import matplotlib.pylab as plot
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
from matplotlib import cm

#GLOBAL PARAMETERS
wavelength = 397e-09
samples = 5000
props = dict(boxstyle='square', facecolor='white', alpha=0.5)

#Auxilliary Functions
def g_two(alpha, gamma, saturation, lamda, ion_distance, loss):
    return (saturation + 1)**2/(saturation + 1 + loss*np.cos(((2*np.pi*ion_distance)/(lamda))*(-np.cos(gamma)-np.cos(alpha))))**2
        
def g_two_3d(alpha, beta, gamma, saturation, lamda, ion_distance, loss):
    return (saturation + 1)**2/(saturation + 1 + loss*np.cos(((2*np.pi*ion_distance)/(lamda))*(-np.sin(gamma)-(np.sin(beta)*np.cos(alpha)))))**2
    
def poissonian(Lamda, k):
    return ((Lamda**k)/(np.math.factorial(k)))*np.exp(-Lamda)

def contrast_loss(values,loss):
    out = []
    for item in values:
        out.append(loss*item+(1-loss)*1)
    #return out
        return values
#auxilliary functions for surface plot
def twoDSampler_grid(center_angle, x_angle, z_angle):
    out = []
    xes = np.linspace(center_angle - (x_angle/2),center_angle + (x_angle/2), num=200)
    for item in xes:
        for item2 in np.linspace(0.5*np.pi - (z_angle/2),0.5*np.pi + (z_angle/2), num=200):
            out.append([item, item2])
    return out
    
def twoDSampler_sample(grid, saturation, lamda, ion_distance, gamma, loss):
    out1 = []
    out2 = []
    out3 = []
    for item in grid:
        out3.append(g_two_3d(item[0], item[1], gamma, saturation, lamda, ion_distance, loss))
        out1.append(item[0])
        out2.append(item[1])
        
    return out1, out2, out3

#Definition of Contrast Loss-Factors
def contrast_loss_time(bins):
    return bins*(-1 * np.exp(-(1/float(bins)))  + 1)
    
def contrast_loss_saturation(saturation_factor):
    return (((saturation_factor+1)**2)/(saturation_factor)**2) - 1

def contrast_loss_spatial(center_angle, gamma,width_of_slit, saturation, ion_distance,loss):
    start = center_angle - np.tan((width_of_slit*0.5)/56)
    end = center_angle + np.tan((width_of_slit*0.5)/56)
    temp = integrate.quad(g_two,start,end,args=(gamma, saturation,wavelength,ion_distance, loss))
    
    temp2 = g_two(np.linspace(start, end, num=samples), gamma,saturation,wavelength,ion_distance, loss)
    
    #xes = np.linspace(start,end,num=1000)
    if __name__ == "__main__":    
        xes = np.linspace(-0.5,0.5,num=samples)
        yes = g_two(xes, gamma,saturation,wavelength,ion_distance, loss)
        plt.fill_between(np.linspace(start,end,num=samples), 0, 5, alpha=0.5, color='green')    
        plt.plot(xes,yes)
        text = (r'Mean_Val: ' + str(np.round(temp[0]/(np.abs(start-end)),decimals=2)))
        plt.text(-0.5, 4, text, None, bbox=props, fontsize=12)
        plt.plot(np.linspace(start, end, num=samples), temp2, color="red")
        
        plt.show() 
    
    return (temp[0]/(np.abs(start-end)))
    
def visualize_gtwo(center_angle, gamma, width_of_slit, saturation, ion_distance,loss):
    plt.clf()
    loss = 0.75* loss
    start = center_angle - np.tan((width_of_slit*0.5)/56)
    end = center_angle + np.tan((width_of_slit*0.5)/56)
    temp = integrate.quad(g_two,start,end,args=(gamma ,saturation,wavelength,ion_distance, loss))
    
    temp2 = g_two(np.linspace(start, end, num=samples), gamma,saturation,wavelength,ion_distance, loss)
    
    sn.set(font_scale=1.15)
    #xes = np.linspace(start,end,num=1000)  
    xes = np.linspace(start-0.08*np.pi,end+0.08*np.pi,num=samples)
    yes = g_two(xes, gamma,saturation,wavelength,ion_distance, loss)
    #plt.fill_between(np.linspace(start,end,num=samples), 0, 6, alpha=0.5, color='green',label="integrated area")
    #plt.fill_between(np.linspace(center_angle-0.3,center_angle+0.3,num=samples), 0, 6, alpha=0.1, color='blue',label="objective area")    
    plt.plot(xes,contrast_loss(yes,loss), label=r'$g^{(2)}(\tau = 0)$')
    plt.title(r'spatial distribution of $g^{(2)}(\tau = 0)$')
    text = (r'Mean_Val: ' + str(np.round((((temp[0]/(np.abs(start-end))-1)*loss)+1),decimals=2)))
    #plt.text(start, 4, text, None, bbox=props, fontsize=12)
    #plt.plot(np.linspace(start, end, num=samples), contrast_loss(temp2,loss), color="red", label="integrated values")
    plt.legend(loc="best", ncol=2)
    plt.xlim([-1.651, -1.52])
    plt.xlabel(r'spatial angle in rad')
    plt.ylabel(r'$g^{(2)}(\vec{r})$ signal for $\tau = 0$')
    plt.savefig('img/integrand.png', dpi=500)
    plt.savefig('img/integrand.pdf')
    plt.show()
    plt.clf()
    plt.plot(np.linspace(-1*np.pi, 1*np.pi, num=samples), contrast_loss(g_two(np.linspace(0, 2*np.pi, num=samples), gamma,saturation,wavelength,ion_distance,loss),loss), color="blue", label="g_two")
    plt.legend(loc="best")
    plt.xlabel(r'spatial angle in $[rad]$')
    plt.ylabel(r'$g^{(2)}$ signal')
    plt.savefig('img/total.png', dpi=500)
    plt.savefig('img/total.pdf')
    plt.show()
    plt.clf()
    if (False):
        #surface
        #0.2*np.pi corresponds with size of objective
        sampled_3d = np.array(twoDSampler_sample(twoDSampler_grid(center_angle, 0.2*np.pi, 0.2*np.pi),saturation,wavelength,ion_distance, gamma, loss))
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        values1 = ax.contourf((sampled_3d[0]).reshape([200, 200]), sampled_3d[1].reshape([200, 200]), sampled_3d[2].reshape([200, 200]),400, cmap=cm.coolwarm, antialiased=False)
        ax.view_init(55, -45)
        # Make a colorbar for the ContourSet returned by the contourf call.
        cbar = plt.colorbar(values1)
        cbar.ax.set_ylabel(r'$g^{(2)}$ signal')
        # Add the contour line levels to the colorbar
        ax.set_xlabel(r'spatial angle in $[rad]$')
        ax.set_ylabel(r'spatial angle in $[rad]$')
        ax.patch.set_facecolor("white")
        fig.savefig('img/view1.png', dpi=500)    
        fig.savefig('img/view1.pdf')
        fig.show()
        
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.set_aspect('equal', adjustable='box')
        #ax2 = fig2.gca(projection='3d')
        #ax2 = fig2.gca().set_aspect('equal', adjustable='box')
        values2 = ax2.contourf((sampled_3d[0]).reshape([200, 200]), sampled_3d[1].reshape([200, 200]), sampled_3d[2].reshape([200, 200]),400, cmap=cm.coolwarm, antialiased=False)
        #ax2.view_init(90, -90)
        cbar = plt.colorbar(values2)
        cbar.ax.set_ylabel(r'$g^{(2)}$ signal')
        ax2.set_xlabel(r'spatial angle in $[rad]$')
        ax2.set_ylabel(r'spatial angle in $[rad]$')
        ax2.patch.set_facecolor("white")
        fig2.savefig('img/view2.png', dpi=500)
        fig2.savefig('img/view2.pdf')
        fig2.show()
        
        fig3 = plt.figure()
        ax3 = fig3.gca(projection='3d')
        values3 = ax3.contourf((sampled_3d[0]).reshape([200, 200]), sampled_3d[1].reshape([200, 200]), sampled_3d[2].reshape([200, 200]),400, cmap=cm.coolwarm, antialiased=False)
        ax3.view_init(20, -90)
        ax3.set_xlabel(r'spatial angle in $[rad]$')
        ax3.set_ylabel(r'spatial angle in $[rad]$')
        cbar = plt.colorbar(values3)
        cbar.ax.set_ylabel(r'$g^{(2)}$ signal')
        ax3.patch.set_facecolor("white")
        fig3.savefig('img/view3.png', dpi=500)
        fig3.savefig('img/view3.pdf')
        fig3.show()

def total_contrast(bins, saturation, center_angle, gamma,width_of_slit, ion_distance,loss):
    return ((contrast_loss_spatial(center_angle, gamma,width_of_slit, saturation, ion_distance,loss)-1)*contrast_loss_time(bins))+1
    
if __name__ == "__main__":
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
    writer = FFMpegWriter(fps=25, metadata=metadata, bitrate=5000)
    
    """
    bins = 3
    xes = np.linspace(0,5,num=1000)
    yes = contrast_loss_saturation(xes)
    yes = yes * contrast_loss_time(bins)
    plt.ylim([0,3])
    plt.plot(xes, yes)
    plt.show()
    plt.clf()
    """
    contrast_loss_spatial(0,5,1,3.12e-6,1,1)
    
    """
    center_angle = -0.0*np.pi
    angle_area = 0.2*np.pi
    
    xes = np.linspace(center_angle - angle_area/2,center_angle + angle_area/2,num=samples)
    yes = g_two(xes,1.2,397e-09,3.4e-6)
    plt.plot(xes,yes)
    
    frames = 500
    
    #VIDEO GEN

    fig = plt.figure() 
    with writer.saving(fig, "img/writer_test.mp4", 350):
        iteration = 1
        bar = progressbar.ProgressBar(max_value=len(np.linspace(3.40e-6,3.70e-6,num=frames)))
        for i in np.linspace(3.00e-6,4.30e-6,num=frames):
            plt.clf()
            plt.ylim([1,3.5])
            xes = np.linspace(center_angle - angle_area/2,center_angle + angle_area/2,num=1000)
            yes = g_two(xes,1.2,397e-09,i)
            plt.plot(xes,yes, color='red') 
            text = (r'Ion-Distance: ' + str(np.round(i*10**6,decimals=3)) + r'$\mu m$')
            plt.text(-0.3, 3, text, None, bbox=props, fontsize=12)
            writer.grab_frame()
            bar.update(iteration)
            iteration = iteration + 1

    plt.clf()
    """