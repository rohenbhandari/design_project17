import os
from math import sin, cos, pi
import numpy as np
from control import *
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import *

import man_ctrl

#X1, Y1 = np.meshgrid(x1, y1)
man_ctrl = man_ctrl.Pendulum(
    .001,
    [0.10, 0.0, 0.0, 0.0],
    30.,
)
#t = np.linspace(0.0, pendulum.end, 1/pendulum.dt)
#print t
data = man_ctrl.integrate()
t = [man_ctrl.end * float(i) / ((man_ctrl.end/man_ctrl.dt) - 1) for i in range(int(man_ctrl.end/man_ctrl.dt))]
#print t

fig = plt.figure(0)
fig.suptitle("Underactuated Link Control with initial conditions")

'''cart_time_line = plt.subplot2grid(
    (12, 12),
    (9, 0),
    colspan=12,
    rowspan=3
)
cart_time_line.axis([
    0,
    10,
    min(data[:,1])*1.1,
    max(data[:,1])*1.1+.1,
])
cart_time_line.set_xlabel('time (s)')
cart_time_line.set_ylabel('theta')
cart_time_line.plot(data[:,0], data[:,1],'r-')'''
'''plt.xlabel('time (s)')
plt.ylabel('angular acceleration of controlled link and theta of uncontrolled link')
plt.plot(t, data[:,0], 'r--')
plt.plot(t, data[:,2], 'b')
grid(True)
legend((r'$\alpha_1$', r'$\theta_2$'))
plt.show()
#plt.plot(data[:,2], data[:,3], 'r--')'''

##########################################################
m=1.0
l=1.0
t1=1.0
t2=1.0

def f(Y, t):
    x1,y1,x2,y2 = Y
    f = [y1,
         ((2./3.)*((t1/(0.5*m*l**2)+(np.sin(x2)*y2*(2*y1+y2))))-(((2./3.)+(np.cos(x2))*((t2/0.5*m*l**2))-np.sin(x2)*y1**2)))/((16./9.)-(np.cos(x2))**2),
         y2,
         -(((2./3.)+np.cos(x2))*((t1/(0.5*m*l**2))+np.sin(x2)*y2*(2*y1+y2))+2*(((5./3.)+np.cos(x2))*(t2/0.5*m*l**2)-np.sin(x2)*y1**2))/((16./9.)-(np.cos(x2))**2)]
    return f

x1 = np.linspace(-40.0, 40.0, 40)
y1 = np.linspace(-40.0, 40.0, 40)
x2 = np.linspace(-40.0, 40.0, 40)
y2 = np.linspace(-40.0, 40.0, 40)
#print x1

X1, Y1 = np.meshgrid(x1, y1)
X2, Y2 = np.meshgrid(x2, y2)
t = 0
u, v = np.zeros(X1.shape), np.zeros(Y1.shape)
NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x1 = X1[i, j]
        y1 = Y1[i, j]
        x2 = X2[i, j]
        y2 = Y2[i, j]
        yprime = f([x1, y1, x2, y2], t)
        #print x1
        #print yprime[0]
        #cprint yprime[1]
        u[i, j] = yprime[2]
        v[i, j] = yprime[3]

Q = plt.quiver(X1, Y1, u, v, color='r')
plt.xlim([-40, 40])
plt.ylim([-40, 40])
plt.show()
###########################################################

'''pendulum_time_line = cart_time_line.twinx()
pendulum_time_line.axis([
    0,
    10,
    min(data[:,3])*1.1-.1,
    max(data[:,3])*1.1
])
pendulum_time_line.set_ylabel('theta (rad)')
pendulum_time_line.plot(data[:,0], data[:,3],'g-')

cart_plot = plt.subplot2grid(
    (12,12),
    (0,0),
    rowspan=8,
    colspan=12
)
cart_plot.axes.get_yaxis().set_visible(False)
plt.show()

time_bar, = cart_time_line.plot([0,0], [10, -10], lw=3)
def draw_point(point):
    time_bar.set_xdata([t, t])
    cart_plot.cla()
    cart_plot.axis([-1.1,.1,-.5,.5])
    cart_plot.plot([point[1]-.1,point[1]+.1],[0,0],'r-',lw=5)
    cart_plot.plot([point[1],point[1]+.4*sin(point[3])],[0,.4*cos(point[3])],'g-', lw=4)
t = 0
fps = 25.
frame_number = 1
for point in data:
    if point[0] >= t + 1./fps or not t:
        draw_point(point)
        t = point[0]
        fig.savefig('/home/rohen/git/nnfl/img/tmp%03d.png' %frame_number)
        frame_number += 1
print os.system("ffmpeg -framerate 25 -i /home/rohen/git/nnfl/img/tmp%03d.png  -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4")'''