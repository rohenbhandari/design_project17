import numpy as np
import control

def vectorfield(w, t, p):
    """
    Defines the differential equations for the coupled manipulator system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1, x2, y2 = w
    t1, t2, m, l = p

    # Create f = (x1',y1',x2',y2'):
    f = [y1,
         ((2./3.)*((t1/(0.5*m*l**2)+(np.sin(x2)*y2*(2*y1+y2))))-(((2./3.)+(np.cos(x2))*((t2/0.5*m*l**2))-np.sin(x2)*y1**2)))/((16./9.)-(np.cos(x2))**2),
         y2,
         -(((2./3.)+np.cos(x2))*((t1/(0.5*m*l**2))+np.sin(x2)*y2*(2*y1+y2))+2*(((5./3.)+np.cos(x2))*(t2/0.5*m*l**2)-np.sin(x2)*y1**2))/((16./9.)-(np.cos(x2))**2)]
    return f

from scipy.integrate import odeint

# Parameter values
# Mass:
m = 1.0
# manipulator constants
l = 1.0
# torques
t1 = 1.0
t2 = 1.0

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
x1 = np.pi
y1 = 3.0
x2 = 0.0
y2 = -1.0

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 250

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [t1, t2, m, l]
w0 = [x1, y1, x2, y2]
# Call the ODE solver.
wsol = odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

with open('coupled_eqn.dat', 'w') as f:
    # Print & save the solution.
    for t1, w1 in zip(t, wsol):
        print >> f, t1, w1[0], w1[1], w1[2], w1[3]

#CONTROL
A11 = (8./3.)*m*l**2
A12 = (2./3.)*m*l**2
A31 = (2./3.)*m*l**2
A32 = (1./3.)*m*l**2
A = np.matrix([
    [0, 1, 0, 0],
    [0, A11, A12, 0],
    [0, 0, 0, 1],
    [0, A31, A32, 0]
])

B1 = 8.1   
B2 = 12.2

B = np.matrix([
    [0],
    [B1],
    [0],
    [B2]
])

#C = np.matrix([0, 0, 0, 0])
#D = np.matrix([0])
Q = np.matrix([
    [10000, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 10000, 0],
    [0, 0, 0, 1]
])

(K, S, E) = control.lqr(A, B, Q, 3.2)
#sys1 = control.ss(A, B, C, D)
pl = control.place(A, B, [0, 0, 0, 0])
print pl
print E

#PLOTTING
from numpy import loadtxt
from pylab import figure, plot, xlabel, grid, hold, legend, title, savefig
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

t, x1, xy, x2, y2 = loadtxt('coupled_eqn.dat', unpack=True)

figure(1, figsize=(6, 4.5))

xlabel('t')
grid(True)
hold(True)
lw = 1

plt.plot(t, x1, 'b', linewidth=lw)
plt.plot(t, x2, 'g', linewidth=lw)
plt.ylim(-15.0, 15.0)

legend((r'$x_1$', r'$x_2$'), prop=FontProperties(size=16))
title('Mass Displacements for the\nManipulator System')
#savefig('two_springs.png', dpi=100)
plt.show()