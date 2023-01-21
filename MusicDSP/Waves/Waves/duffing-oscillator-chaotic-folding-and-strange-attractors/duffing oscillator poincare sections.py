#!/usr/bin/env python3
#
# Program poincare:
# computes the poincare section for the duffing equation with intersection of 
# theta=omega*t=0 plane with a defined phase shift and plots this as v against x

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


# sets coefficients of the duffing equation 
beta  = -1.0
alpha = 0.25
gamma = 1.0
omega = 1.0
Fe    = 0.41

# sets a phase shift by a factor phase*2pi, adjust this to see how the poincare section changes
# and see charcteristic stretching and folding
phase = 1.0

# defines the ode's for the duffing equation
def duffing(x, t):
    return [x[1],-beta*x[0] - 2*alpha*x[1] - gamma*x[0]**3 \
                  + Fe*np.cos(omega*t + phase*2*np.pi)]


# Calculates the values of x and v taken 5000 times at each
# value of omega*t=2pi using odeint
x = []
v = []
fig, ax = plt.subplots(figsize=(6, 6))
t=np.linspace(0, 5000*(2*np.pi), 25000000)
duff = odeint(duffing, [1, 0], t)
x = [duff[5000*i, 0] for i in range(5000)]
v = [duff[5000*i, 1] for i in range(5000)]

# plot of v against x for the poincare setions
ax.scatter(v, x, color = 'b', s=0.1)
plt.xlabel('v', fontsize=16)
plt.ylabel('x',labelpad=-10, fontsize=16)
plt.tick_params(labelsize=16)
phase2=phase*2
plt.title('The Poincare section with phase shift = %1.1f pi' %phase2)
plt.show()