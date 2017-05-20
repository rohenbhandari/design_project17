import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pend(y, t, b, c):
	theta, x = y
	dydt = [x, b*x + c]
	return dydt

theta2 = 0/4.
alpha = 1.

b = (-(4./3.)*np.sin(theta2) + np.sin(2*theta2)*alpha) / ((16./9.) - (np.cos(theta2))**2)
c = (-(8./3.) -4*np.sin(theta2)*alpha**2 + 2*np.cos(theta2) - np.sin(2*theta2)*alpha**2 + 2*np.cos(theta2) + (20./3.) + 4*np.cos(theta2)) / (((16./9.) - (np.cos(theta2))**2))

y0 = [np.pi/2. - 0.1, 0.0]
t = np.linspace(0, 10, 101)

sol = odeint(pend, y0, t, args=(b, c))

#print t
#print size(sol[:, 0])
#print size(sol[:, 1])

plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.xlabel('t')
plt.grid()
plt.show()