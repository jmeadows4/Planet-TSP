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

init_cond = np.zeros(4)
t_min = 0
t_max = 0
next_planet = None
dist_argmin = -1
Sun_mass = 1
G = 4*np.pi**2
mu = Sun_mass * G
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

dist_x = 0
dist_y = 0
dist_total = 100000
num_calls = 0
total_calls = 0
num_paths = 0

# This function will use the global variables t_min/t_max, init_cond, and next_planet.
# The variables cannot be passed in because minimize varies the parameters
def roc_to_planet_dist(init_vel):

    global dist_argmin
    global dist_x
    global dist_y
    global dist_total
    global num_calls
    global total_calls
    global num_paths
    num_calls += 1
    total_calls += 1
    print("current minimize calls = ", num_calls)
    print("total calls = ", total_calls)
    print("num paths = ", num_paths)
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
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return dist_total


def plot(start_p, end_p, sol):
    global dist_argmin
    sol_x = sol.y[0, :]
    sol_x = sol_x[0:dist_argmin+1]
    sol_y = sol.y[1, :]
    sol_y = sol_y[0:dist_argmin+1]
#    plt.ylim(-7, 7)
#    plt.xlim(-7, 7)
    t_arr = np.linspace(0, max(start_p.p, end_p.p), 100)
    plt.plot(sol_x, sol_y, '*g', label = 'Rocket Path', markersize = 4)
    plt.plot(sol_x[0], sol_y[0], '*g', label = 'Rocket Start', markersize = 8)
    plt.plot(end_p.get_x(t_arr), end_p.get_y(t_arr), 'or', label = end_p.name, markersize = 1 )
    plt.plot(start_p.get_x(t_arr), start_p.get_y(t_arr), 'ob', label = start_p.name, markersize = 1 )
    plt.plot(end_p.get_x(0), end_p.get_y(0), 'ob', label = "Starting point"+end_p.name, markersize = 7)
    plt.plot(end_p.get_x(sol.t[dist_argmin]), end_p.get_y(sol.t[dist_argmin]), 'or', label = "Closest point"+end_p.name, markersize = 7)
    plt.plot(sol_x[dist_argmin], sol_y[dist_argmin], '*b', label = "Closest Rocket point", markersize = 9)

    plt.plot(0, 0, 'o', color = 'orange', markersize = 7)
    plt.legend()
    plt.xlabel("Au")
    plt.ylabel("Au")
    plt.show()

def Plot_Energy(sol):
    v_roc_x = sol.y[2, :]
    v_roc_y = sol.y[3, :]
    v_roc = np.sqrt(v_roc_x**2 + v_roc_y**2)

    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]
    r_roc = np.sqrt(sol_x**2 + sol_y**2)
    dist_next = np.sqrt((sol_x - next_planet.get_x(sol.t))**2 + (sol_y - next_planet.get_y(sol.t))**2)

    KE = .5*v_roc**2
    PE_sun = -1/r_roc
    TE = KE - PE_sun

    plt.figure()
    plt.plot(sol.t, TE)
    plt.title("Energy vs Time")
    plt.xlabel("Time [yrs]")
    plt.ylabel("Total Energy [$M_{\odot} Au^{2}yrs^{-2}$]")
    plt.show()


#Code for the Hohmann Transfer with functions for x and y delta v
def delta_v1_x(time, cur_planet, next_planet):
    r1 = cur_planet.get_r(time)
    r2 = next_planet.get_r(time)
    theta = cur_planet.w*time + cur_planet.pi_factor*np.pi
    v1_x = np.sqrt(mu/r1)*(np.sqrt(2*r2/(r1+r2))-1)
    return v1_x * (-1 * np.sin(theta))

def delta_v1_y(time, cur_planet, next_planet):
    r1 = cur_planet.get_r(time)
    r2 = next_planet.get_r(time)
    theta = cur_planet.w*time + cur_planet.pi_factor*np.pi
    v1_y = np.sqrt(mu/r1)*(np.sqrt(2*r2/(r1+r2))-1)
    return v1_y * np.cos(theta)

def get_final_velocities(sol, dist_argmin):
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]
    delta_x = sol_x[dist_argmin] - sol_x[dist_argmin-1]
    delta_y = sol_y[dist_argmin] - sol_y[dist_argmin-1]
    delta_t = sol.t[dist_argmin] - sol.t[dist_argmin-1]
    vx_final = delta_x / delta_t
    vy_final = delta_y / delta_t
    return [vx_final, vy_final]

#change these to the planets you want to go to/from
cur_planet = earth
next_planet = venus
# distance factor has to change for different planets? .01 works for
# larger starting planets(like Jupiter), but .001 is better for smaller planets(Earth, Mars)
distance_factor = .001
init_cond[0] = cur_planet.get_x(0) + distance_factor
init_cond[1] = cur_planet.get_y(0) + distance_factor
#minimum and maximum time to get to the planet. Might need to adjust depending on the planets
t_min = 0
t_max = 1

#This is code using the root function to find a "shortest" path.
#Comment it out if you are doing Hohmann Transfer below

#create array to save all possible velocities in 
vel_time_arr = np.zeros((1, 3))
a = np.array([["V_x", "V_y", "t"]]) #creates header for array
vel_time_arr = np.append(a, vel_time_arr, axis=0)
vel_time_arr = np.delete(vel_time_arr, 1, 0)

for i in range(-7, 7):
    for j in range(-7, 7):
        v0 = [i, j]
        minimize(roc_to_planet_dist, v0, method = "L-BFGS-B",
#                 options = {'maxfun': 20},
                 bounds =((-10, 10), (-10, 10)))
        num_calls = 0
        if dist_total < 1e-3:
            print("found a path")
            num_paths += 1
            sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)
            vel_time_arr = np.append(vel_time_arr, [[sol.y[2][0], sol.y[3][0], sol.t[dist_argmin]]], axis=0)
#            plot(cur_planet, next_planet, sol)
#            #CHANGE YOUR DIRECTORY TO WHERE YOU WANT TO SAVE THE FIGURE
#            plt.savefig('/home/jmeadows4/Documents/PHYS498/Planet-TSP/earth_mars_path/fig'
#                        +str(num_paths)+'.png')
#            plt.close("all")


print("all done!")
#creates date and time string for file name
now = dt.datetime.now()
t_str = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = "poss_vels_"+t_str+".txt"

#saves vel array as txt file
np.savetxt(filename, vel_time_arr, fmt = '%s', delimiter = ', ') #saves vel_time_arr to text file

#i = 7
#v0 = [i/3, -i/1]
#init_vels = minimize(roc_to_planet_dist, v0, method = "L-BFGS-B", bounds =((-i, i), (-i, i)), maxfun = 20)
#sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)
#plot(cur_planet, next_planet, sol)


#plt.show()

#sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)
#plot(cur_planet, next_planet, sol)



###############################################################################


####This is code for the Hohmann Transfer. Comment it out if you are running root/plot above ####
#init_cond[2] = earth.get_tang_vel_x(0) + delta_v1_x(0, cur_planet, next_planet)
#init_cond[3] = earth.get_tang_vel_y(0) + delta_v1_y(0, cur_planet, next_planet)
#sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)
#print(sol)
#plot(cur_planet, next_planet, sol)
###############################################################################



########Old code that isn't useful as of right now ##########

# def delta_v2_x(time, cur_planet, next_planet):
#     r1 = cur_planet.get_r(time)
#     r2 = next_planet.get_r(time)
#     theta = cur_planet.w*time + cur_planet.pi_factor*np.pi
#     v2_x = np.sqrt(mu/r1)*(1 - np.sqrt(2*r1/(r1+r2)))
#     return v2_x * (-1 * np.sin(theta))
#
# def delta_v2_y(time, cur_planet, next_planet):
#     r1 = cur_planet.get_r(time)
#     r2 = next_planet.get_r(time)
#     theta = cur_planet.w*time + cur_planet.pi_factor*np.pi
#     v2_y = np.sqrt(mu/r1)*(1 - np.sqrt(2*r1/(r1+r2)))
#     return v2_y * np.cos(theta)


