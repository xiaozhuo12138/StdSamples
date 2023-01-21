from scipy.integrate import odeint
import numpy as np

mu = 8.6

def vanderpol1(X, t):
    x = X[0]
    y = X[1]
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

def vanderpol2(X, t):
    x = X[0]
    y = X[1]
    dxdt = mu * (x - 1./3.*x**3 - y)
    dydt = x / mu
    return [dxdt, dydt]


X0 = [1, 2]
t = np.linspace(0, 40, 250)

sol = odeint(vanderpol1, X0, t)

import matplotlib.pyplot as plt
x = sol[:, 0]
y = sol[:, 1]

plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x')
plt.legend(('x', 'y'))
plt.title(r'x vs t at $\mu$ = 5')
# plt.savefig('vanderpol_cmu1.png')
plt.show()
# phase portrait
# plt.figure()
# plt.plot(x,y)
# plt.plot(x[0], y[0], 'ro')
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.savefig('vanderpol_cmu2.png')
# plt.show()