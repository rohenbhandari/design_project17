import numpy as np 
import matplotlib.pyplot as plt 

def f(Y, t):
	y1, y2 = Y
	theta2 = 0/4.
	alpha = 1.
	b = (-(4./3.)*np.sin(theta2) + np.sin(2*theta2)*alpha) / ((16./9.) - (np.cos(theta2))**2)
	c = (-(8./3.) -4*np.sin(theta2)*alpha**2 + 2*np.cos(theta2) - np.sin(2*theta2)*alpha**2 + 2*np.cos(theta2) + (20./3.) + 4*np.cos(theta2)) / (((16./9.) - (np.cos(theta2))**2))
	return[y2, b*y1 + c]

y1 = np.linspace(-40.0, 40.0, 40)
y2 = np.linspace(-40.0, 40.0, 40)

Y1, Y2 = np.meshgrid(y1, y2)
t = 0
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
print Y1.shape
NI, NJ = Y1.shape

for i in range(NI):
	for j in range(NJ):
		x = Y1[i, j]
		y = Y2[i, j]
		yprime = f([x, y], t)
		print x
		print y
		#print yprime[0]
		#print yprime[1]
		u[i, j] = yprime[0]
		v[i, j] = yprime[1]

Q = plt.quiver(Y1, Y2, u, v, color='r')
plt.xlim([-40, 40])
plt.ylim([-40, 40])
plt.show()