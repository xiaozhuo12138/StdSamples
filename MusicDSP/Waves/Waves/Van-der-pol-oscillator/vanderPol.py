import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu = 5

def van_der_pol_oscillator_deriv1(x, t):
    nx0 = x[1]
    nx1 = -mu * (x[0] ** 2.0 - 1.0) * x[1] - x[0]
    res = np.array([nx0, nx1])
    return res


def van_der_pol_oscillator_deriv2(x, t):
    nx0 = mu * (x[0] - (1.0/3.0)*x[0]**3 - x[1] )
    nx1 = x[0]/mu
    res = np.array([nx0, nx1])
    return res

ts = np.linspace(0.0, 50.0, 500)

xs = odeint(van_der_pol_oscillator_deriv1, [0.2, 0.2], ts)
plt.plot(xs[:,0], xs[:,1])
xs = odeint(van_der_pol_oscillator_deriv1, [-3.0, -3.0], ts)
plt.plot(xs[:,0], xs[:,1])
xs = odeint(van_der_pol_oscillator_deriv1, [4.0, 4.0], ts)
plt.plot(xs[:,0], xs[:,1])
plt.gca().set_aspect('equal')
plt.title('Limit Cycle (mu=5)')
plt.xlabel('x')
plt.ylabel(r'$y = \.x$')
# plt.savefig('vanderpol_oscillator.png')
plt.show() 