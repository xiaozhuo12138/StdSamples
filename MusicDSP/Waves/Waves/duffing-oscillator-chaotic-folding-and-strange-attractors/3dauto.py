#!/usr/bin/env python3
#
# Program 3Dauto: 
# modelling the duffing equation as a 3D autonomous system producing v-x phase portraits and the solution x(t)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Duffing parameters and initial conditions
beta  = -1.0
alpha = -0.1
Fe    = 1
x0, v0, theta0 = 0, 1, 1.05


# Sets range in t and step number
trange, n = 30, 10000

# Sets up the ODE's for the problem
def Duffing(X, t, alpha, beta, Fe):
    """The Duffing equations"""

    x, v, theta = X
    dx = v
    dv = -beta * x - 2*alpha*v - x**3 + Fe*np.cos(theta)
    dtheta = -Fe * np.sin(theta)
    return dx, dv, dtheta

# Solves the ODE's at each time step using odeint
t = np.linspace(0, trange, n)
duff = odeint(Duffing, (x0, v0, theta0), t, args=(alpha, beta, Fe))
x, v, theta = duff.T

# When plotting comment out plt.clf and uncomment plt.show for the one you want

# Phase portrait
plt.plot(x, v, "b-", lw=0.5)
plt.xlabel('x', fontsize=15)
plt.ylabel('v', fontsize=15,labelpad=-10)
plt.tick_params(labelsize=15)
plt.title('Phase portrait for the 3D autonomous system')
plt.show()
plt.clf()

# Solution x(t)
plt.plot(t, x)
plt.xlabel('t', fontsize=15)
plt.ylabel('x(t)', fontsize=15)
plt.xlim(0,trange)
plt.ylim(-5,5)
plt.title('Solution x(t) for the 3D autonomous system')
plt.show()