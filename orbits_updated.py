import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
plt.close("all")

class Planet:

    def __init__(self, m, ec, ax, factor):
        self.mass = m
        self.ecc = ec
        self.a = ax
        self.b = ax * np.sqrt(1 - ec**2)
        self.c = ax * ec
        self.p = ax**(3/2)
        self.w = 2*np.pi/.24
        self.pi_factor = factor
        self.next = None
    def get_x(self, time):
        return self.a * np.cos(self.w*time - self.pi_factor*np.pi) + self.c
    def get_y(self, time):
        return self.b * np.cos(self.w*time - self.pi_factor*np.pi)

#####################   Globals   ############################################
Sun_mass = 1
G = 4*np.pi**2                                      #in AU #Newtons constant
n_t = 500

mercury = Planet(1.652e-7, .2056, .3870, 1/2)
venus = Planet(2.447e-6, .0068, .7219, 1/3)
earth = Planet(3.003e-6, .017, 1.00001423349, 0)
mars = Planet(3.213e-7, .0934, 1.52408586388, 7/8)
jupiter = Planet(9.543e-4, .0484, 5.2073, 11/8)
saturn = Planet(2.857e-4, .0542, 9.5590, 0)
uranus = Planet(4.365e-5, .0472, 19.1848, 13/16)
neptune = Planet(5.145e-5, .0086, 30.0806, 9/16)

planets = [earth, mars, mercury, venus, jupiter, saturn, uranus, neptune]
init_cond = np.zeros(4)
t_min = -1
t_max = -1
next_planet = None
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

dist_x = .01
dist_y = .01

#this function will use the global variables t_min/t_max, init_cond, and next_planet
def get_planet_dist(init_vel):

    global dist_x
    global dist_y

    init_cond[2] = init_vel[0]
    init_cond[3] = init_vel[1]

    sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-4)
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]

    dist_next = np.sqrt((sol_x - next_planet.get_x(sol.t))**2 + (sol_y - next_planet.get_y(sol.t))**2)
    dist_argmin = np.argmin(dist_next)
    dist_x = np.abs(sol_x[dist_argmin] - next_planet.get_x(sol.t[dist_argmin]))
    dist_y = np.abs(sol_y[dist_argmin] - next_planet.get_y(sol.t[dist_argmin]))
    print("Distance in x, y to planet: ", dist_x, dist_y)
    print("Correct initial velocities: ", init_vel)
    print("Time taken to reach planet: ", sol.t[dist_argmin] )
    print("Minimum distance index: ", dist_argmin)
    return [dist_x + 1e-3, dist_y + 1e-3]


init_cond[0] = earth.get_x(0) + .01
init_cond[1] = earth.get_y(0) + .01
v0 = [2, -6.188]
t_min = 0
t_max = .5

#The main code
for i in range(1,3):

    #before each root call, we will need to update init_cond, tmin/tmax, and next_planet
    next_planet = planets[i]
    #wait.... root returns the optimized velocities, not the minimum distance....
    print("Starting next planet adventure :)")
    next_vel = root(get_planet_dist, v0)

    v0 = next_vel.x # .x returns the optimized values
    init_cond[0] = dist_x
    init_cond[1] = dist_y
