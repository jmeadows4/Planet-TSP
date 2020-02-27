#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:07:35 2020

@author: dcanderson
"""
#imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
plt.close("all")

#variables
Sun_mass = 1
G = 4*np.pi**2 
mu = G*Sun_mass
pi = np.pi
t=0

Earth_mass = 3.003e-6                               #in Solar masses #Mass of Earth
ecc_earth = 0.017                                   #eccentricity of planet rotation
a_earth = 1.00001423349                             #Au #Semi major axis of planet/sun system
b_earth = a_earth*np.sqrt(1 - ecc_earth**2)         #Au #Semi minor axis of planet/sun system
c_earth = a_earth*ecc_earth                         #Au #distance from center of orbit to sun
p_earth = a_earth**(3/2)                            #period of orbit #estimation through Keplers 3rd
w_earth = 2*np.pi/1                                 #angular freq in rads/earth years

Mars_mass = 3.213e-7                                #in Solar masses #Mass of Earth
ecc_mars = 0.0934                                   #eccentricity of planet rotation
a_mars = 1.52408586388                              #Au #Semi major axis of planet/sun system
b_mars = a_mars*np.sqrt(1 - ecc_mars**2)            #Au #Semi minor axis of planet/sun system
c_mars = a_mars*ecc_mars                            #Au-1 #distance from center of orbit to sun
p_mars = a_mars**(3/2)                              #period of orbit #estimation through Keplers 3rd
w_mars = 2*np.pi/1.88                               #angular freq in rads/earth years
    
    
#Functions-------------------------------------------------------------
def x_earth(time):  
    #earth's x position as function of time
    return a_earth*np.cos(w_earth*time) + c_earth

def y_earth(time):  
    #earth's y position as function of time
    return b_earth*np.sin(w_earth*time)

def x_mars(time):  
    #mars's x position as function of time
    return a_mars*np.cos(w_mars*time + 7*np.pi/8) + c_mars

def y_mars(time): 
    #mars's y position as function of time
    return b_mars*np.sin(w_mars*time + 7*np.pi/8)

def Rocket_man(time, state):                        #I think it's going to be a long, long time
    #Solves diff eq with time inputed, and initial conditions (states)
    #returns array of x velocities, y velocities, x accelerations and y accelerations of rocket, respectively
    
    x0 = state[0]                                   #initial position of rocket in x
    y0 = state[1]                                   #initial position of rocket in y

    v1 = state[2]                                   #v_x_rocket #initial x velocity of rocket
    u1 = state[3]                                   #v_y_rocket #initial y velocity of rocket
    
    rhs = np.zeros(len(state)) #right hand side array (solutions)
    
    #distances from rocket to planet
    rRoEa = np.sqrt((x0 - x_earth(time))**2 + (y0 - y_earth(time))**2)
    rRoMa = np.sqrt((x0 - x_mars(time))**2 + (y0 - y_mars(time))**2)
    rRoSu = np.sqrt(x0**2 + y0**2) #distance from rocket to sun (with sun at origin)

    rhs[0] = v1                                       #initial x velocity of rocket
        
    rhs[1] = u1                                       #iniitial y velocity of rocket
    
    #calculates x acceleration due to gravity from all bodies
    rhs[2] = (G*(Earth_mass*(x_earth(time) - x0)/rRoEa**3
       + Mars_mass*(x_mars(time) - x0)/rRoMa**3
       + Sun_mass*(-x0)/rRoSu**3))
    
    #calculates y acceleration due to gravity from all bodies
    rhs[3] = (G*(Earth_mass*(y_earth(time) - y0)/rRoEa**3
       + Mars_mass*(y_mars(time) - y0)/rRoMa**3
       + Sun_mass*(-y0)/rRoSu**3))
    
    return rhs

r1 = np.sqrt(x_earth(t)**2 + y_earth(t)**2)
r2 = np.sqrt(x_mars(t)**2 + y_mars(t)**2)

def delta_v1(t): #to get from orbit 1 to orbit 2 (see wiki)
    v1 = np.sqrt(mu/r1)*(np.sqrt(2*r1/(r1 + r2))-1)
    return v1

def delta_v2(t): #to get from orbit 2 to orbit 3 (see wiki) may not use right away
    v2 = np.sqrt(mu/r2)*(1 - np.sqrt(2*r1/(r1 + r2)))
    return v2
    
def ang_alignment(t): #has to do with starting vel change when bodies are properly aligned
    a = pi*(1 - (1/4)*(np.sqrt(2*(r1/r2 + 1)**3)))
    return a

#But for now, just looking to get to correct radius, so will not use ang alignment as of yet
#will use angular velocity of planet to get tangential velocity so we can use that velocity to calculate the direction of our new vel
def tan_vel(w): #just to get from earth to mars, will have to generalize the radii (r1, r2)
    tan_v = w*r1
    return tan_v

init_cond = np.zeros(4)
init_cond[0] = x_earth(0)
init_cond[1] = y_earth(0)
init_cond[2] = 0
init_cond[3] = tan_vel(w_earth) + delta_v1

sol = solve_ivp(Rocket_man, init_cond)

