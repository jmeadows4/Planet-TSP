#Dr. Nelson notes:
#First, plot the paths in a nice way.
#Then, pick a velocity, and try the next planet.
#use cmap?

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from math import log, sin, cos
plt.close("all")

class Planet:
    def __init__(self, m, name, ec, ax, factor, w_factor):
        self.mass = m
        self.name = name
        self.ecc = ec
        self.a = ax
        self.b = ax * np.sqrt(1 - ec**2)
        self.c = ax * ec
        self.p = ax**(3/2)
        self.w = 2*np.pi/w_factor
        self.pi_factor = factor
    def get_x(self, time):
        return self.a * np.cos(self.w*time + self.pi_factor*np.pi) + self.c
    def get_y(self, time):
        return self.b * np.sin(self.w*time + self.pi_factor*np.pi)
    def get_r(self, time):
        return np.sqrt(self.get_x(time)**2 + self.get_y(time)**2)
    def get_tang_vel_x(self, time):
        theta = self.w*time + self.pi_factor*np.pi
        v_tang_x = -self.w * self.get_r(time) * sin(theta)
        return v_tang_x
    def get_tang_vel_y(self, time):
        theta = self.w*time + self.pi_factor*np.pi
        v_tang_y = self.w * self.get_r(time) * cos(theta)
        return v_tang_y
    def get_roche():
        q = self.mass
        return get_r(self, time) * (.49 * q**(2/3)) / (.6 * q**(2/3) + log(1 + q**(1/3)))
#####################   Globals   #############################################

mercury = Planet(1.652e-7, "Mercury", .2056, .3870, -1/2, .24 )
venus = Planet(2.447e-6, "Venus", .0068, .7219, 1/3, .616)
earth = Planet(3.003e-6, "Earth", .017, 1.00001423349, 0, 1)
mars = Planet(3.213e-7, "Mars", .0934, 1.52408586388, 7/8, 1.88)
jupiter = Planet(9.543e-4, "Jupiter", .0484, 5.2073, 11/8, 12)
saturn = Planet(2.857e-4, "Saturn", .0542, 9.5590, 0, 29)
uranus = Planet(4.365e-5, "Uranus", .0472, 19.1848, 13/16, 84)
neptune = Planet(5.145e-5, "Neptune", .0086, 30.0806, 9/16, 165)
planets = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]
init_cond = np.zeros(4)
t_min = 0
t_max = 0
next_planet = None
dist_argmin = -1
Sun_mass = 1
G = 4*np.pi**2
mu = Sun_mass * G
dist_total = 100000
total_calls = 0
num_paths = 0
###############################################################################

def Rocket_man(time, state):                        #I think it's going to be a long, long time
    #Solves diff eq with time inputed, and initial conditions (states)
    #returns array of x velocities, y velocities, x accelerations and y accelerations of rocket, respectively

    x0 = state[0]                                   #initial position of rocket in x
    y0 = state[1]                                   #initial position of rocket in y

    v1 = state[2]                                   #v_x_rocket #initial x velocity of rocket
    u1 = state[3]                                   #v_y_rocket #initial y velocity of rocket

    rhs = np.zeros(len(state)) #right hand side array (solutions)

    #distances from rocket to planet
    rRoMe = np.sqrt((x0 - mercury.get_x(time))**2 + (y0 - mercury.get_y(time))**2)
    rRoVe = np.sqrt((x0 - venus.get_x(time))**2 + (y0 - venus.get_y(time))**2)
    rRoEa = np.sqrt((x0 - earth.get_x(time))**2 + (y0 - earth.get_y(time))**2)
    rRoMa = np.sqrt((x0 - mars.get_x(time))**2 + (y0 - mars.get_y(time))**2)
    rRoJu = np.sqrt((x0 - jupiter.get_x(time))**2 + (y0 - jupiter.get_y(time))**2)
    rRoSa = np.sqrt((x0 - saturn.get_x(time))**2 + (y0 - saturn.get_y(time))**2)
    rRoUr = np.sqrt((x0 - uranus.get_x(time))**2 + (y0 - uranus.get_y(time))**2)
    rRoNe = np.sqrt((x0 - neptune.get_x(time))**2 + (y0 - neptune.get_y(time))**2)
    rRoSu = np.sqrt(x0**2 + y0**2) #distance from rocket to sun (with sun at origin)

    rhs[0] = v1                                       #initial x velocity of rocket

    rhs[1] = u1                                       #iniitial y velocity of rocket

    #calculates x acceleration due to gravity from all bodies
    rhs[2] = (G*(mercury.mass*(mercury.get_x(time) - x0)/rRoMe**3
       + venus.mass*(venus.get_x(time) - x0)/rRoVe**3
       + earth.mass*(earth.get_x(time) - x0)/rRoEa**3
       + mars.mass*(mars.get_x(time) - x0)/rRoMa**3
       + jupiter.mass*(jupiter.get_x(time) - x0)/rRoJu**3
       + saturn.mass*(saturn.get_x(time) - x0)/rRoSa**3
       + uranus.mass*(uranus.get_x(time) - x0)/rRoUr**3
       + neptune.mass*(neptune.get_x(time) - x0)/rRoNe**3
       + Sun_mass*(-x0)/rRoSu**3))

    #calculates y acceleration due to gravity from all bodies
    rhs[3] = (G*(mercury.mass*(mercury.get_y(time) - y0)/rRoMe**3
       + venus.mass*(venus.get_y(time) - y0)/rRoVe**3
       + earth.mass*(earth.get_y(time) - y0)/rRoEa**3
       + mars.mass*(mars.get_y(time) - y0)/rRoMa**3
       + jupiter.mass*(jupiter.get_y(time) - y0)/rRoJu**3
       + saturn.mass*(saturn.get_y(time) - y0)/rRoSa**3
       + uranus.mass*(uranus.get_y(time) - y0)/rRoUr**3
       + neptune.mass*(neptune.get_y(time) - y0)/rRoNe**3
       + Sun_mass*(-y0)/rRoSu**3))

    return rhs

# This function will use the global variables t_min/t_max, init_cond, and next_planet.
# The variables cannot be passed in because minimize varies the parameters
def roc_to_planet_dist(init_vel):
    #need to specify global variables so that the function does not create local variables
    global dist_argmin, dist_total, total_calls
    total_calls += 1

    init_cond[2] = init_vel[0]
    init_cond[3] = init_vel[1]
    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)

    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]
    dist_next = np.sqrt((sol_x - next_planet.get_x(sol.t))**2 + (sol_y - next_planet.get_y(sol.t))**2)
    dist_argmin = np.argmin(dist_next)
    dist_x = np.abs(sol_x[dist_argmin] - next_planet.get_x(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - next_planet.get_y(sol.t[dist_argmin]))
    dist_total = np.sqrt(dist_x**2 + dist_y**2)
    print("total calls = ", total_calls)
    print("num paths = ", num_paths)
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    return dist_total


def plot(start_p, end_p):
    t_arr = np.linspace(0, max(start_p.p, end_p.p), 100)
    #plot planet orbits
    plt.xlim(-1.5, 2.5)
    plt.plot(sol_x, sol_y, "*g", markersize = 4)
    plt.plot(start_p.get_x(t_arr), start_p.get_y(t_arr), 'ob', label = start_p.name, markersize = 1 )
    plt.plot(end_p.get_x(t_arr), end_p.get_y(t_arr), 'or', label = end_p.name, markersize = 1 )

    plt.plot(0, 0, 'o', color = 'orange', markersize = 7)
    plt.legend()
    plt.xlabel("Au")
    plt.ylabel("Au")
    plt.show()

def plot_gif(start_p, end_p, sol):
    #Rocket path
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]
    dist_next = np.sqrt((sol_x - next_planet.get_x(sol.t))**2 + (sol_y - next_planet.get_y(sol.t))**2)
    dist_argmin = np.argmin(dist_next)
    t_arr = np.linspace(0, max(start_p.p, end_p.p), 100)
    t_a = np.linspace(sol.t.min(), sol.t.max(), len(sol.t))
    for i in range(len(sol.t)):
        plt.xlim(-1.5, 2.5)
        plt.ylim(-1.5, 1.5)
        new_sol_x = sol.y[0, 0:i]
        new_sol_y = sol.y[1, 0:i]
        plt.plot(new_sol_x, new_sol_y, "*g", markersize = 4)

        plt.plot(end_p.get_x(t_a[i]), end_p.get_y(t_a[i]), "or", markersize = 7)
        plt.plot(start_p.get_x(t_a[i]), start_p.get_y(t_a[i]), 'ob', label = start_p.name, markersize = 7 )

        plt.plot(0, 0, 'o', color = 'orange', markersize = 8)
        plt.legend()
        plt.xlabel("Au")
        plt.ylabel("Au")
        plt.savefig('/home/jmeadows4/Documents/PHYS498/Planet-TSP/gif_earth_mars_better/fig'
                    +str(i)+'.png')
        plt.close("all")



#change these to the planets you want to go to/from
cur_planet = earth
next_planet = mars

# distance factor has to change for different planets? .01 works for
# larger starting planets(like Jupiter), but .001 is better for smaller planets(Earth, Mars)
distance_factor = .001

init_cond[0] = cur_planet.get_x(0) + distance_factor
init_cond[1] = cur_planet.get_y(0) + distance_factor
init_cond[2] = -5.15937227
init_cond[3] = .96192575
#minimum and maximum time to get to the planet. Might need to adjust depending on the planets
t_min = 0
t_max = 1

sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)

plot_gif(cur_planet, next_planet, sol)
