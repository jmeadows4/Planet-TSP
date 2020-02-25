import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
plt.close("all")

class Planet:

    def __init__(self, m, name, ec, ax, factor):
        self.name = name
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
        return self.b * np.sin(self.w*time - self.pi_factor*np.pi)
    def get_r(self, time):
        return np.sqrt(self.get_x(self, time)**2 + self.get_y(self, time)**2)

#####################   Globals   ############################################
Sun_mass = 1
G = 4*np.pi**2

mercury = Planet(1.652e-7, "Mercury", .2056, .3870, 1/2)
venus = Planet(2.447e-6, "Venus", .0068, .7219, 1/3)
earth = Planet(3.003e-6, "Earth", .017, 1.00001423349, 0)
mars = Planet(3.213e-7, "Mars", .0934, 1.52408586388, 7/8)
jupiter = Planet(9.543e-4, "Jupiter", .0484, 5.2073, 11/8)
saturn = Planet(2.857e-4, "Saturn", .0542, 9.5590, 0)
uranus = Planet(4.365e-5, "Uranus", .0472, 19.1848, 13/16)
neptune = Planet(5.145e-5, "Neptune", .0086, 30.0806, 9/16)

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

#this function will use the global variables t_min/t_max, init_cond, and next_planet
def get_planet_dist(init_vel):

    #return a large value if it tries a large velocity
    if abs(init_vel[0]) > 20 or abs(init_vel[1]) > 20 :
        return [100000, 100000]

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



def plot(start_p, end_p, sol):
    sol_x = sol.y[0, :]
    sol_y = sol.y[1, :]
    plt.plot(sol_x, sol_y, '*g', label = 'Rocket Path', markersize = 4)
    plt.plot(start_p.get_x(sol.t), start_p.get_y(sol.t), 'ob', label = start_p.name, markersize = 2)
    plt.plot(end_p.get_x(sol.t), end_p.get_y(sol.t), 'or', label = end_p.name, markersize = 2)
    plt.plot(0, 0, 'o', color = 'orange', markersize = 7)
    plt.legend()
    plt.xlabel("Au")
    plt.ylabel("Au")
    plt.show()


#uses root to get a best initial velocity guess
cur_planet = earth
next_planet = mars
init_cond[0] = cur_planet.get_x(0) + .01
init_cond[1] = cur_planet.get_y(0) + .01
t_min = 0
t_max = .5
v0 = [2, -6]
next_vel = root(get_planet_dist, v0)

#uses the final velocity from root for one last solve_ivp. I only do this to get sol
init_cond[2] = next_vel.x[0]
init_cond[3] = next_vel.x[1]
sol = solve_ivp(Rocket_man, (t_min, t_max), init_cond, rtol = 1e-8)
if sol.status < 0:
    print("uh oh, error!")

plot(cur_planet, next_planet, sol)
