#!/usr/bin/env python3
#
# Program phasepace: 
# produces phase-space diagrams of the unforced duffing oscillator (Fe = 0)

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pylab as pl

# Duffing equation constants where alpha should be varied
beta  = -1.0
alpha = -0.2
gamma = 1.0
# Defines the ODEs of the duffing equation
def duffing(x, t):
    return [x[1],-beta*x[0] - 2*alpha*x[1] - gamma*x[0]**3]

# Next section computes trajectories in a range of timesteps defined by steps
# and a range of inital conditions set by init

# Forward trajectories
steps = np.linspace(0, 10, 500)
init = np.linspace(-4, 4, 2)
for x in init:
    for v in init:
        x0 = [x, v]
# odeint is used to solve the ODE's        
        duff = odeint(duffing, x0, steps)
        plt.plot(duff[:,0], duff[:,1], "b-")

# Backwards trajectories
steps = np.linspace(0, -10, 500)
init = np.linspace(-4, 4, 2)
for x in init:
    for v in init:
        x0 = [x, v]
        duff = odeint(duffing, x0, steps)
        plt.plot(duff[:,0], duff[:,1], "r-")

# Sets axes labels
plt.xlabel("$\dot{x}$", fontsize=15)
plt.ylabel("$\dot{v}$", fontsize=15, labelpad=-10)
plt.tick_params(labelsize=15)
plt.xlim(-7, 7)
plt.ylim(-20, 20)

# Plot the vectorfield with arrows to see nullclines etc.
X, V = np.mgrid[-7:7:10j, -20:20:10j]
xdot=V
vdot=-beta*X - gamma*X**3 - 2*alpha*V
pl.quiver(X, V, xdot, vdot, color = 'c')
plt.title('Phase Space Diagram with $\\alpha$ = %1.1f' %alpha)
plt.show()