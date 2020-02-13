#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:09:58 2019

@author: dcanderson
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
plt.close("all")

#******************************************************************************
'''-------------------------------variables---------------------------------'''

Sun_mass = 1
G = 4*np.pi**2                                      #in AU #Newtons constant
n_t = 500

Mercury_mass = 1.652e-7                             #in Solar masses
ecc_merc = .2056                                    #eccentricity of planet rotation
a_merc = .3870                                      #Au #Semi major axis of planet/sun system
b_merc = a_merc*np.sqrt(1 - ecc_merc**2)            #Au #Semi minor axis of planet/sun system
c_merc = a_merc*ecc_merc                            #Au #distance from center of orbit to sun
p_merc = a_merc**(3/2)                              #period of orbit #through Keplers 3rd
w_merc = 2*np.pi/.24                                #angular freq in rads/earth years

Venus_mass = 2.447e-6                               #in Solar masses #Mass of
ecc_venus = .0068                                   #eccentricity of planet rotation
a_venus = .7219                                     #Au #Semi major axis of planet/sun system
b_venus = a_venus*np.sqrt(1 - ecc_venus**2)         #Au #Semi minor axis of planet/sun system
c_venus = a_venus*ecc_venus                         #Au #distance from center of orbit to sun
p_venus = a_venus**(3/2)                            #period of orbit #through Keplers 3rd
w_venus = 2*np.pi/.616                              #angular freq in rads/earth years

Earth_mass = 3.003e-6                               #in Solar masses #Mass of Earth
ecc_earth = 0.017                                   #eccentricity of planet rotation
a_earth = 1.00001423349                             #Au #Semi major axis of planet/sun system
b_earth = a_earth*np.sqrt(1 - ecc_earth**2)         #Au #Semi minor axis of planet/sun system
c_earth = a_earth*ecc_earth                         #Au #distance from center of orbit to sun
p_earth = a_earth**(3/2)                            #period of orbit #through Keplers 3rd
w_earth = 2*np.pi/1                                 #angular freq in rads/earth years

Mars_mass = 3.213e-7                                #in Solar masses #Mass of Earth
ecc_mars = 0.0934                                   #eccentricity of planet rotation
a_mars = 1.52408586388                              #Au #Semi major axis of planet/sun system
b_mars = a_mars*np.sqrt(1 - ecc_mars**2)            #Au #Semi minor axis of planet/sun system
c_mars = a_mars*ecc_mars                            #Au-1 #distance from center of orbit to sun
p_mars = a_mars**(3/2)                              #period of orbit #through Keplers 3rd
w_mars = 2*np.pi/1.88                                #angular freq in rads/earth years

Jupiter_mass = 9.543e-4                             #in Solar masses #Mass of
ecc_jup = .0484                                     #eccentricity of planet rotation
a_jup = 5.2073                                      #Au #Semi major axis of planet/sun system
b_jup = a_jup*np.sqrt(1 - ecc_jup**2)               #Au #Semi minor axis of planet/sun system
c_jup = a_jup*ecc_jup                               #Au #distance from center of orbit to sun
p_jup = a_jup**(3/2)                                #period of orbit #through Keplers 3rd
w_jup = 2*np.pi/12                                  #angular freq in rads/earth years

Saturn_mass = 2.857e-4                              #in Solar masses #Mass of
ecc_sat = .0542                                     #eccentricity of planet rotation
a_sat = 9.5590                                      #Au #Semi major axis of planet/sun system
b_sat = a_sat*np.sqrt(1 - ecc_sat**2)               #Au #Semi minor axis of planet/sun system
c_sat = a_sat*ecc_jup                               #Au #distance from center of orbit to sun
p_sat = a_sat**(3/2)                                #period of orbit #through Keplers 3rd
w_sat = 2*np.pi/29                                  #angular freq in rads/earth years

Uranus_mass = 4.365e-5                              #in Solar masses #Mass of
ecc_uran = .0472                                    #eccentricity of planet rotation
a_uran = 19.1848                                    #Au #Semi major axis of planet/sun system
b_uran = a_uran*np.sqrt(1 - ecc_uran**2)            #Au #Semi minor axis of planet/sun system
c_uran = a_uran*ecc_uran                            #Au #distance from center of orbit to sun
p_uran = a_uran**(3/2)                              #period of orbit #through Keplers 3rd
w_uran = 2*np.pi/84                                 #angular freq in rads/earth years

Neptune_mass = 5.146e-5                             #in Solar masses #Mass of
ecc_nep = .0086                                     #eccentricity of planet rotation
a_nep = 30.0806                                     #Au #Semi major axis of planet/sun system
b_nep = a_nep*np.sqrt(1 - ecc_nep**2)               #Au #Semi minor axis of planet/sun system
c_nep = a_nep*ecc_nep                               #Au #distance from center of orbit to sun
p_nep = a_nep**(3/2)                                #period of orbit #through Keplers 3rd
w_nep = 2*np.pi/165                                 #angular freq in rads/earth years

#******************************************************************************
'''-------------------------------functions---------------------------------'''


#******************************************************************************
'''--------------------------------Mercury----------------------------------'''

def x_merc(time):
    #mercury's x position as function of time
    return a_merc*np.cos(w_merc*time - np.pi/2) + c_merc

def y_merc(time):
    #mercury's y position as function of time
    return b_merc*np.sin(w_merc*time - np.pi/2)

def roc_merc_dist(init_vel):
    #finds distance from mercury to rocket (used to find roots -> min vels)
    #input must be initial velocities (so root can minimize them), outputs distance to planet with wiggle room

    #initial condition array
    init_cond = np.zeros(4) #x, y position will change depending on where rocket is relative to last visited planet
    init_cond[0] = sol_x_ven[11] #will get positions from outputted dictionary through solve_ivp (all named sol_[Planet])
    init_cond[1] = sol_y_ven[11]
    init_cond[2] = init_vel[0] #we will guess x, y vels, then run through root to find min vels to get rocket to next planet, and update to min vels
    init_cond[3] = init_vel[1]

    t_min = 0.7
    t_max = .8

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-4)
    sol_x = sol.y[0, :] #array of x positions for rocket (to be plotted)
    sol_y = sol.y[1, :] #array of y positions for rocket (to be plotted)

    #when put into "root", "root" will try multiple init_vels until it finds a min
    #below prints info every run through so we can see progress
    dist_merc = np.sqrt((sol_x -x_merc(sol.t))**2 + (sol_y -y_merc(sol.t))**2)
    dist_argmin = np.argmin(dist_merc)
    dist_x = np.abs(sol_x[dist_argmin] - x_merc(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - y_merc(sol.t[dist_argmin]))
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x + 1e-3, dist_y + 1e-3]

#******************************************************************************
'''---------------------------------Venus-----------------------------------'''

def x_ven(time):
    #venus's x position as function of time
    return a_venus*np.cos(w_venus*time + np.pi/3) + c_venus

def y_ven(time):
    #venus's y position as function of time
    return b_venus*np.sin(w_venus*time + np.pi/3)

def roc_ven_dist(init_vel):
    #finds distance from venus to rocket (used to find roots -> min vels)
    #input must be initial velocities (so root can minimize them), outputs distance to planet with wiggle room

    init_cond = np.zeros(4) #see roc_merc_dist
    init_cond[0] = sol_x_mars[11]
    init_cond[1] = sol_y_mars[11]
    init_cond[2] = init_vel[0]
    init_cond[3] = init_vel[1]

    t_min = .5
    t_max = 1

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-4)
    sol_x = sol.y[0, :] #array of x positions for rocket (to be plotted)
    sol_y = sol.y[1, :] #array of y positions for rocket (to be plotted)

    #when put into "root", "root" will try multiple init_vels until it finds a min
    #below prints info every run through so we can see progress
    dist_ven = np.sqrt((sol_x -x_ven(sol.t))**2 + (sol_y -y_ven(sol.t))**2)
    dist_argmin = np.argmin(dist_ven)
    dist_x = np.abs(sol_x[dist_argmin] - x_ven(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - y_ven(sol.t[dist_argmin]))
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x + 1e-3, dist_y + 1e-3]

#******************************************************************************
'''---------------------------------Earth-----------------------------------'''

def x_earth(time):
    #earth's x position as function of time
    return a_earth*np.cos(w_earth*time) + c_earth

def y_earth(time):
    #earth's y position as function of time
    return b_earth*np.sin(w_earth*time)

#******************************************************************************
'''---------------------------------Mars------------------------------------'''

def x_mars(time):
    #mars's x position as function of time
    return a_mars*np.cos(w_mars*time + 7*np.pi/8) + c_mars

def y_mars(time):
    #mars's y position as function of time
    return b_mars*np.sin(w_mars*time + 7*np.pi/8)

def roc_mars_dist(init_vel):
    #finds distance from mars to rocket (used to find roots -> min vels)
    #input must be initial velocities (so root can minimize them), outputs distance to planet with wiggle room

    init_cond = np.zeros(4) #see roc_merc_dist
    init_cond[0] = x_earth(0) + .01
    init_cond[1] = y_earth(0) + .01
    init_cond[2] = init_vel[0]
    init_cond[3] = init_vel[1]

    t_min = 0
    t_max = .5

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]

    #when put into "root", "root" will try multiple init_vels until it finds a min
    #below prints info every run through so we can see progress
    dist_mars = np.sqrt((sol_x -x_mars(sol.t))**2 + (sol_y -y_mars(sol.t))**2)
    dist_argmin = np.argmin(dist_mars)
    dist_x = np.abs(sol_x[dist_argmin] - x_mars(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - y_mars(sol.t[dist_argmin]))
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x + 1e-3, dist_y + 1e-3]

#******************************************************************************
'''-------------------------------Jupiter-----------------------------------'''

def x_jup(time):
    #jupiters's x position as function of time
    return a_jup*np.cos(w_jup*time + 11*np.pi/8) + c_jup

def y_jup(time):
    #jupiter's y position as function of time
    return b_jup*np.sin(w_jup*time + 11*np.pi/8)

def roc_jup_dist(init_vel):
    #finds distance from jupiter to rocket (used to find roots -> min vels)
    #input must be initial velocities (so root can minimize them), outputs distance to planet with wiggle room

    init_cond = np.zeros(4) #see roc_merc_dist
    init_cond[0] = x_earth(0) + .01
    init_cond[1] = y_earth(0) + .01
    init_cond[2] = init_vel[0]
    init_cond[3] = init_vel[1]

    t_min = .8
    t_max = 2.8

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-4)
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]

    #when put into "root", "root" will try multiple init_vels until it finds a min
    #below prints info every run through so we can see progress
    dist_merc = np.sqrt((sol_x -x_merc(sol.t))**2 + (sol_y -y_merc(sol.t))**2)
    dist_argmin = np.argmin(dist_merc)
    dist_x = np.abs(sol_x[dist_argmin] - x_merc(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - y_merc(sol.t[dist_argmin]))
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x + 1e-3, dist_y + 1e-3]

#******************************************************************************
'''-------------------------------Saturn------------------------------------'''

def x_sat(time):
    #saturn's x position as function of time
    return a_sat*np.cos(w_sat*time) + c_sat

def y_sat(time):
    #saturn's y position as function of time
    return b_sat*np.sin(w_sat*time)

#finds distance from planet to rocket******************************************
    #used with root runction to find min vels
def roc_sat_dist(init_vel):
    #finds distance from saturn to rocket (used to find roots -> min vels)
    #input must be initial velocities (so root can minimize them), outputs distance to planet with wiggle room

    init_cond = np.zeros(4) #see roc_merc_dist
    init_cond[0] = x_earth(0) + .01
    init_cond[1] = y_earth(0) + .01
    init_cond[2] = init_vel[0]
    init_cond[3] = init_vel[1]

    t_min = 2.8
    t_max = 4.52

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-4)
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]

    #when put into "root", "root" will try multiple init_vels until it finds a min
    #below prints info every run through so we can see progress
    dist_merc = np.sqrt((sol_x -x_merc(sol.t))**2 + (sol_y -y_merc(sol.t))**2)
    dist_argmin = np.argmin(dist_merc)
    dist_x = np.abs(sol_x[dist_argmin] - x_merc(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - y_merc(sol.t[dist_argmin]))
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x + 1e-3, dist_y + 1e-3]

#******************************************************************************
'''-------------------------------Uranus------------------------------------'''

def x_uran(time):
    #uranus's x position as function of time
    return a_uran*np.cos(w_uran*time + 13*np.pi/16) + c_uran

def y_uran(time):
    #uranus's y position as function of time
    return b_uran*np.sin(w_uran*time + 13*np.pi/16)

def roc_uran_dist(init_vel):
    #finds distance from uranus to rocket (used to find roots -> min vels)
    #input must be initial velocities (so root can minimize them), outputs distance to planet with wiggle room

    init_cond = np.zeros(4) #see roc_merc_dist
    init_cond[0] = x_earth(0) + .01
    init_cond[1] = y_earth(0) + .01
    init_cond[2] = init_vel[0]
    init_cond[3] = init_vel[1]

    t_min = 0
    t_max = 1

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-4)
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]

    #when put into "root", "root" will try multiple init_vels until it finds a min
    #below prints info every run through so we can see progress
    dist_uran = np.sqrt((sol_x -x_uran(sol.t))**2 + (sol_y -y_uran(sol.t))**2)
    dist_argmin = np.argmin(dist_uran)
    dist_x = np.abs(sol_x[dist_argmin] - x_uran(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - y_uran(sol.t[dist_argmin]))
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x + 1e-3, dist_y + 1e-3]

#******************************************************************************
'''-------------------------------Neptune-----------------------------------'''

def x_nep(time):
    #neptunes's x position as function of time
    return a_nep*np.cos(w_nep*time + 9*np.pi/16) + c_nep

def y_nep(time):
    #neptune's y position as function of time
    return b_nep*np.sin(w_nep*time + 9*np.pi/16)

def roc_nep_dist(init_vel):
    #finds distance from neptune to rocket (used to find roots -> min vels)
    #input must be initial velocities (so root can minimize them), outputs distance to planet with wiggle room

    init_cond = np.zeros(4)  #see roc_merc_dist
    init_cond[0] = x_earth(0) + .01
    init_cond[1] = y_earth(0) + .01
    init_cond[2] = init_vel[0] #initial velocity in x
    init_cond[3] = init_vel[1] #initial velocity in y

    t_min = 4.58
    t_max = 7.3

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-5)
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]

    #when put into "root", "root" will try multiple init_vels until it finds a min
    #below prints info every run through so we can see progress
    dist_nep = np.sqrt((sol_x -x_nep(sol.t))**2 + (sol_y -y_nep(sol.t))**2)
    dist_argmin = np.argmin(dist_nep)
    dist_x = sol_x[dist_argmin] - x_nep(sol.t[dist_argmin])
    dist_y = sol_y[dist_argmin] - y_nep(sol.t[dist_argmin])
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x, dist_y]

#******************************************************************************
'''--------------------------------Rocket-----------------------------------'''

def Rocket_man(time, state):                        #I think it's going to be a long, long time
    #Solves diff eq with time inputed, and initial conditions (states)
    #returns array of x velocities, y velocities, x accelerations and y accelerations of rocket, respectively

    x0 = state[0]                                   #initial position of rocket in x
    y0 = state[1]                                   #initial position of rocket in y

    v1 = state[2]                                   #v_x_rocket #initial x velocity of rocket
    u1 = state[3]                                   #v_y_rocket #initial y velocity of rocket

    rhs = np.zeros(len(state)) #right hand side array (solutions)

    #distances from rocket to planet
    rRoMe = np.sqrt((x0 - x_merc(time))**2 + (y0 - y_merc(time))**2)
    rRoVe = np.sqrt((x0 - x_ven(time))**2 + (y0 - y_ven(time))**2)
    rRoEa = np.sqrt((x0 - x_earth(time))**2 + (y0 - y_earth(time))**2)
    rRoMa = np.sqrt((x0 - x_mars(time))**2 + (y0 - y_mars(time))**2)
    rRoJu = np.sqrt((x0 - x_jup(time))**2 + (y0 - y_jup(time))**2)
    rRoSa = np.sqrt((x0 - x_sat(time))**2 + (y0 - y_sat(time))**2)
    rRoUr = np.sqrt((x0 - x_uran(time))**2 + (y0 - y_uran(time))**2)
    rRoNe = np.sqrt((x0 - x_nep(time))**2 + (y0 - y_nep(time))**2)
    rRoSu = np.sqrt(x0**2 + y0**2) #distance from rocket to sun (with sun at origin)

    rhs[0] = v1                                       #initial x velocity of rocket

    rhs[1] = u1                                       #iniitial y velocity of rocket

    #calculates x acceleration due to gravity from all bodies
    rhs[2] = (G*(Mercury_mass*(x_merc(time) - x0)/rRoMe**3
       + Venus_mass*(x_ven(time) - x0)/rRoVe**3
       + Earth_mass*(x_earth(time) - x0)/rRoEa**3
       + Mars_mass*(x_mars(time) - x0)/rRoMa**3
       + Jupiter_mass*(x_jup(time) - x0)/rRoJu**3
       + Saturn_mass*(x_sat(time) - x0)/rRoSa**3
       + Uranus_mass*(x_uran(time) - x0)/rRoUr**3
       + Neptune_mass*(x_nep(time) - x0)/rRoNe**3
       + Sun_mass*(-x0)/rRoSu**3))

    #calculates y acceleration due to gravity from all bodies
    rhs[3] = (G*(Mercury_mass*(y_merc(time) - y0)/rRoMe**3
       + Venus_mass*(y_ven(time) - y0)/rRoVe**3
       + Earth_mass*(y_earth(time) - y0)/rRoEa**3
       + Mars_mass*(y_mars(time) - y0)/rRoMa**3
       + Jupiter_mass*(y_jup(time) - y0)/rRoJu**3
       + Saturn_mass*(y_sat(time) - y0)/rRoSa**3
       + Uranus_mass*(y_uran(time) - y0)/rRoUr**3
       + Neptune_mass*(y_nep(time) - y0)/rRoNe**3
       + Sun_mass*(-y0)/rRoSu**3))

    return rhs

#Finds minimum distance between the planet and rocket to be plotted
def min_dist(x_plan, y_plan, sol_x, sol_y, sol_t):
    dist_planet = np.sqrt((sol_x -x_plan(sol_t))**2 + (sol_y -y_plan(sol_t))**2)
    dist_argmin = np.argmin(dist_planet)
    dist_x = sol_x[dist_argmin] - x_nep(sol_t[dist_argmin])
    dist_y = sol_y[dist_argmin] - y_nep(sol_t[dist_argmin])
    dist_r = np.sqrt(dist_x**2 + dist_y**2)
    return dist_r

#Plots and saves figures for animation*****************************************
def plot_anim_in(sol_t, sol_x, sol_y, plot_number, ylim_lo, ylim_up, xlim_lo, xlim_up):

    for i in range(len(sol_t)):

        plt.close("all")
        plt.ylim(ylim_lo, ylim_up)
        plt.xlim(xlim_lo, xlim_up)

        if ylim_up <= 3:
            plt.plot(x_merc(sol_t[i]), y_merc(sol_t[i]), 'o',
                     label = "Mercury", markersize = 5)
            plt.plot(x_ven(sol_t[i]), y_ven(sol_t[i]), 'o',
                     label = "Venus", markersize = 5)
            plt.plot(x_earth(sol_t[i]), y_earth(sol_t[i]), 'ob',
                     label = "Earth", markersize = 5)
            plt.plot(x_mars(sol_t[i]), y_mars(sol_t[i]), 'or',
                     label = "Mars", markersize = 5)
            plt.plot(0, 0, 'o', color = 'orange', markersize = 14)
        if ylim_up > 3:
            plt.plot(x_merc(sol_t[i]), y_merc(sol_t[i]), 'o',
                     label = "Mercury", markersize = 3)
            plt.plot(x_ven(sol_t[i]), y_ven(sol_t[i]), 'o',
                     label = "Venus", markersize = 3)
            plt.plot(x_earth(sol_t[i]), y_earth(sol_t[i]), 'ob',
                     label = "Earth", markersize = 3)
            plt.plot(x_mars(sol_t[i]), y_mars(sol_t[i]), 'or',
                     label = "Mars", markersize = 3)
            plt.plot(x_jup(sol_t[i]), y_jup(sol_t[i]), 'o',
                     label = "Jupiter", markersize = 5)
            plt.plot(x_sat(sol_t[i]), y_sat(sol_t[i]), 'o',
                     label = "Saturn", markersize = 5)
            plt.plot(0, 0, 'o', color = 'orange', markersize = 12)
        if ylim_up > 6:
            plt.plot(x_merc(sol_t[i]), y_merc(sol_t[i]), 'o',
                     label = "Mercury", markersize = 1)
            plt.plot(x_ven(sol_t[i]), y_ven(sol_t[i]), 'o',
                     label = "Venus", markersize = 1)
            plt.plot(x_earth(sol_t[i]), y_earth(sol_t[i]), 'ob',
                     label = "Earth", markersize = 1)
            plt.plot(x_mars(sol_t[i]), y_mars(sol_t[i]), 'or',
                     label = "Mars", markersize = 1)
            plt.plot(x_jup(sol_t[i]), y_jup(sol_t[i]), 'o',
                     label = "Jupiter", markersize = 3)
            plt.plot(x_sat(sol_t[i]), y_sat(sol_t[i]), 'o',
                     label = "Saturn", markersize = 3)
            plt.plot(x_uran(sol_t[i]), y_uran(sol_t[i]), 'o',
                     label = "Uranus", markersize = 3)
            plt.plot(x_nep(sol_t[i]), y_nep(sol_t[i]), 'o',
                     label = "Neptune", markersize = 3)
            plt.plot(0, 0, 'o', color = 'orange', markersize = 9)

        plt.plot(sol_x[i], sol_y[i], '^g', label = "Rocket Path", markersize = 2)

        plt.savefig('plot'+str(i + plot_number)+'.png')

#******************************************************************************
'''--------------------------Initial Conditions-----------------------------'''
#******************************************************************************
#might not need these init conds if we can get root(the minimization function) and roc_planet_dist's to work together
'''**********************to get to mars from earth**************************'''

init_cond_mars = np.zeros(4)
init_cond_mars[0] = x_earth(0) + .01                     #initial x position of rocket
init_cond_mars[1] = y_earth(0) + .01                     #initial y position of rocket
init_cond_mars[2] = 2                                    #initial v_x of rocket  #au/year
init_cond_mars[3] = -6.188                               #initial v_y of rocket  #au/year

#time stuff
t_min_mars = 0
t_max_mars = .5
time_mars = np.linspace(t_min_mars, t_max_mars, n_t)

sol_mars = solve_ivp(Rocket_man, (t_min_mars, t_max_mars), init_cond_mars, rtol = 1e-8)
sol_x_mars = sol_mars.y[0, :]
sol_y_mars = sol_mars.y[1, :]

#Plots Distance to planet vs time**********************************************
#plt.figure()
#dist_mars_plot = np.sqrt((sol_x_mars -x_mars(sol_mars.t))**2
#                         + (sol_y_mars -y_mars(sol_mars.t))**2)
#plt.plot(sol_mars.t, dist_mars_plot)
#plt.xlabel("time")
#plt.ylabel("Distance between Mars and rocket in Au")
#plt.title("Distance to Mars vs Time")
#plt.show()

#******************************************************************************
'''**********************to get to venus from mars**************************'''

init_cond_ven = np.zeros(4)
init_cond_ven[0] = sol_x_mars[29]                        #initial x position of rocket
init_cond_ven[1] = sol_y_mars[29]                        #initial y position of rocket
init_cond_ven[2] = -3.15                                 #initial v_x of rocket  #au/year
init_cond_ven[3] = 8.4                                   #initial v_y of rocket  #au/year

t_min_ven = .5
t_max_ven = .7
time_ven = np.linspace(t_min_ven, t_max_ven, n_t)

sol_ven = solve_ivp(Rocket_man, (t_min_ven, t_max_ven), init_cond_ven, rtol = 1e-8)
sol_x_ven = sol_ven.y[0, :]
sol_y_ven = sol_ven.y[1, :]

#Plots Distance to planet vs time**********************************************
#plt.figure()
#dist_ven_plot = np.sqrt((sol_x_ven -x_ven(sol_ven.t))**2
#                        + (sol_y_ven -y_ven(sol_ven.t))**2)
#plt.plot(sol_ven.t, dist_ven_plot)
#plt.xlabel("Time in years")
#plt.ylabel("Distance between Venus and rocket in Au")
#plt.title("Distance to Venus vs Time")
#plt.show()

#******************************************************************************
'''*********************to get to mercury from venus************************'''

init_cond_merc = np.zeros(4)
init_cond_merc[0] = sol_x_ven[24]                        #initial x position of rocket
init_cond_merc[1] = sol_y_ven[24]                        #initial y position of rocket
init_cond_merc[2] = 7.07                                 #initial v_x of rocket  #au/year
init_cond_merc[3] = -.15                                 #initial v_y of rocket  #au/year

t_min_merc = 0.7
t_max_merc = .8
time_merc = np.linspace(t_min_merc, t_max_merc, n_t)

sol_merc = solve_ivp(Rocket_man, (t_min_merc, t_max_merc), init_cond_merc, rtol = 1e-8)
sol_x_merc = sol_merc.y[0, :]
sol_y_merc = sol_merc.y[1, :]

#Plots Distance to planet vs time**********************************************
#plt.figure()
#dist_merc_plot = np.sqrt((sol_x_merc -x_merc(sol_merc.t))**2
#                         + (sol_y_merc -y_merc(sol_merc.t))**2)
#plt.plot(sol_merc.t, dist_merc_plot)
#plt.xlabel("Time in years")
#plt.ylabel("Distance between Mercury and rocket in Au")
#plt.title("Distance to Mercury vs Time")
#plt.show()

#******************************************************************************
'''*********************to get to Jupiter from Mercury**********************'''

init_cond_jup = np.zeros(4)
init_cond_jup[0] = sol_x_merc[31]                        #initial x position of rocket
init_cond_jup[1] = sol_y_merc[31]                        #initial y position of rocket
init_cond_jup[2] = 12.549                                #initial v_x of rocket  #au/year
init_cond_jup[3] = -.875                                 #initial v_y of rocket  #au/year

t_min_jup = 0.8
t_max_jup = 2.8
time_jup = np.linspace(t_min_jup, t_max_jup, n_t)

sol_jup = solve_ivp(Rocket_man, (t_min_jup, t_max_jup), init_cond_jup, rtol = 1e-8)
sol_x_jup = sol_jup.y[0, :]
sol_y_jup = sol_jup.y[1, :]

#Plots Distance to planet vs time**********************************************
#plt.figure()
#dist_jup_plot = np.sqrt((sol_x_jup -x_jup(sol_jup.t))**2
#                        + (sol_y_jup -y_jup(sol_jup.t))**2)
#plt.plot(sol_jup.t, dist_jup_plot)
#plt.xlabel("Time in years")
#plt.ylabel("Distance between Jupiter and rocket in Au")
#plt.title("Distance to Jupiter vs Time")
#plt.show()

#******************************************************************************
'''*********************to get to Saturn from Jupiter***********************'''

init_cond_sat = np.zeros(4)
init_cond_sat[0] = sol_x_jup[71]                         #initial x position of rocket
init_cond_sat[1] = sol_y_jup[71]                         #initial y position of rocket
init_cond_sat[2] = -9.44                                 #initial v_x of rocket  #au/year
init_cond_sat[3] = 2.409                                 #initial v_y of rocket  #au/year

t_min_sat = 2.8
t_max_sat = 4.59
time_sat = np.linspace(t_min_sat, t_max_sat, n_t)

sol_sat = solve_ivp(Rocket_man, (t_min_sat, t_max_sat), init_cond_sat, rtol = 1e-8)
sol_x_sat = sol_sat.y[0, :]
sol_y_sat = sol_sat.y[1, :]

#Plots Distance to planet vs time**********************************************
#plt.figure()
#dist_sat_plot = np.sqrt((sol_x_sat -x_sat(sol_sat.t))**2
#                        + (sol_y_sat -y_sat(sol_sat.t))**2)
#plt.plot(sol_sat.t, dist_sat_plot)
#plt.xlabel("Time in years")
#plt.ylabel("Distance between Saturn and rocket in Au")
#plt.title("Distance to Saturn vs Time")
#plt.show()

#******************************************************************************
'''*********************to get to Uranus from Saturn************************'''

init_cond_uran = np.zeros(4)
init_cond_uran[0] = sol_x_sat[99]                        #initial x position of rocket
init_cond_uran[1] = sol_y_sat[99]                        #initial y position of rocket
init_cond_uran[2] = -5.2                                 #initial v_x of rocket  #au/year
init_cond_uran[3] = -8.594                               #initial v_y of rocket  #au/year

t_min_uran = 4.59
t_max_uran = 7.29
time_uran = np.linspace(t_min_uran, t_max_uran, n_t)

sol_uran = solve_ivp(Rocket_man, (t_min_uran, t_max_uran), init_cond_uran, rtol = 1e-8)
sol_x_uran = sol_uran.y[0, :]
sol_y_uran = sol_uran.y[1, :]

#Plots Distance to planet vs time**********************************************
#plt.figure()
#dist_uran_plot = np.sqrt((sol_x_uran -x_uran(sol_uran.t))**2
#                         + (sol_y_uran -y_uran(sol_uran.t))**2)
#plt.plot(sol_uran.t, dist_uran_plot)
#plt.xlabel("Time in years")
#plt.ylabel("Distance between Uranus and rocket in Au")
#plt.title("Distance to Uranus vs Time")
#plt.show()

#******************************************************************************
'''*********************to get to Neptune from Uranus***********************'''

init_cond_nep = np.zeros(4)
init_cond_nep[0] = sol_x_uran[79]                        #initial x position of rocket
init_cond_nep[1] = sol_y_uran[79]                        #initial y position of rocket
init_cond_nep[2] = 0.987                                 #initial v_x of rocket  #au/year
init_cond_nep[3] = 11.2                                  #initial v_y of rocket  #au/year

t_min_nep = 7.29
t_max_nep = 9.496
time_nep = np.linspace(t_min_nep, t_max_nep, n_t)

sol_nep = solve_ivp(Rocket_man, (t_min_nep, t_max_nep), init_cond_nep, rtol = 1e-8)
sol_x_nep = sol_nep.y[0, :]
sol_y_nep = sol_nep.y[1, :]

#Plots Distance to planet vs time**********************************************
#plt.figure()
#dist_nep_plot = np.sqrt((sol_x_nep -x_nep(sol_nep.t))**2
#                         + (sol_y_nep -y_nep(sol_nep.t))**2)
#plt.plot(sol_nep.t, dist_nep_plot)
#plt.xlabel("Time in years")
#plt.ylabel("Distance between Neptune and rocket in Au")
#plt.title("Distance to Neptune vs Time")
#
#plt.show()

#******************************************************************************
'''---------------------------------Plot------------------------------------'''
#calls animation plot function*************************************************
#plot_anim_in(sol_mars.t, sol_x_mars, sol_y_mars, 0, -3, 3, -3, 3)
#plot_anim_in(sol_ven.t, sol_x_ven, sol_y_ven, 30, -3, 3, -3, 3)
#plot_anim_in(sol_merc.t, sol_x_merc, sol_y_merc, 55, -3, 3, -3, 3)
#plot_anim_in(sol_jup.t, sol_x_jup, sol_y_jup, 87, -6, 6, -6, 6)
#plot_anim_in(sol_sat.t, sol_x_sat, sol_y_sat, 159, -9, 9, -9, 9)
#plot_anim_in(sol_uran.t, sol_x_uran, sol_y_uran, 259, -30, 30, -20, 20)
#plot_anim_in(sol_nep.t, sol_x_nep, sol_y_nep, 339, -30, 30, -20, 20)

#to refrence while looking for initial velocities******************************
#can put in different time values and see rocket path, was using to check how close rocket was to planet
#now just a nice plot
t_min = 0
t_max = 50
plt.figure()
time = np.linspace(t_min, t_max, n_t)
plt.ylim(-10000, 10000)
plt.xlim(-10000, 10000)

plt.plot(sol_x_nep, sol_y_nep, '*g', label = "Rocket Path", markersize = 4)
plt.plot(sol_x_ven, sol_y_ven, '*g', markersize = 4)
plt.plot(sol_x_merc, sol_y_merc, '*g', markersize = 4)
plt.scatter(sol_x_merc, sol_y_merc, marker = 'o')
plt.plot(x_merc(time)*1e+4, y_merc(time)*1e+4, 'o', label = "Mercury", markersize = .326)
plt.plot(x_ven(time)*1e+4, y_ven(time)*1e+4, 'o', label = "Venus", markersize = .86)
plt.plot(x_earth(time)*1e+4, y_earth(time)*1e+4, 'ob', label = "Earth", markersize = .86)
plt.plot(x_mars(time)*1e+4, y_mars(time)*1e+4, 'or', label = "Mars", markersize = .86)
plt.plot(x_jup(time)*1e+4, y_jup(time)*1e+4, 'o', label = "Jupiter", markersize = 9.34)
plt.plot(x_sat(time)*1e+4, y_sat(time)*1e+4, 'o', label = "Saturn", markersize = 7.78)
plt.plot(x_uran(time)*1e+4, y_uran(time)*1e+4, 'o', label = "Uranus", markersize = 3.40)
plt.plot(x_nep(time)*1e+4, y_nep(time)*1e+4, 'o', label = "Neptune", markersize = 3.29)
plt.plot(0, 0, 'o', color = 'orange', markersize = 92)
plt.legend()
plt.xlabel("10,000 = 1 Au")
plt.ylabel("10,000 = 1 Au")
plt.show()

#important stuff, also where things didn't seem to be working
#Finds roots to solve for correct min velocities*******************************
#v0_mars = np.array([2, -6.188]) #5, -5 #-3, -3
#min_dist_mars = root(roc_mars_dist, v0_mars)

#v0_ven = np.array([5.1, .03])
#min_dist_ven = root(roc_ven_dist, v0_ven)

#v0_merc = np.array([-1, -4])
#min_dist_merc = root(roc_merc_dist, v0_merc)

#v0_jup = np.array([12, -.87])
#min_dist_jup = root(roc_jup_dist, v0_jup)

#v0_sat = np.array([-9, -2])
#min_dist_sat = root(roc_sat_dist, v0_sat)

#v0_uran = np.array([-6, -8])
#min_dist_merc = root(roc_merc_dist, v0_merc)

#v0_nep = np.array([-1, -4])
#min_dist_merc = root(roc_merc_dist, v0_merc)
