from scipy.integrate import odeint
import numpy as np 
import matplotlib.pyplot as plt 

def vectorfield(w, t, p):
	x1, y1, x2, y2 = w
	m1, m2, l1, l2 = p
	f = [y1, (l1*y1 + m1*y2 + x1), y2, (m2*y1 + l2*y2 - x2)]
	return f

m1 = 1.0
m2 = 1.5

l1 = 1.
l2 = 1.

x1 = 0.8
x2 = 1.
y1 = 0.0
y2 = 0.0

abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 250

t = [stoptime*float(i) / (numpoints - 1) for i in range(numpoints)]

p = [m1, m2, l1, l2]
w0 = [x1, y2, x2, y2]

wsol = odeint(vectorfield, w0, t, args=(p,), atol=abserr, rtol=relerr)
for t1, w1 in zip(t, wsol):
	print t1, w1[0], w1[1], w1[2], w1[3]

plt.plot(t, x1, 'b')
plt.plot(t, x2, 'g')
plt.show()