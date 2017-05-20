from math import sin, cos, pi
from numpy import matrix, array
from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np 
import skfuzzy as fuzz
from skfuzzy import control as ctrl 

M = .6  # mass of cart+pendulum
m = 1.0  # mass of pendulum
Km = 2  # motor torque constant
Kg = .01  # gear ratio
R = 6 # armiture resistance
r = .01  # drive radius
K1 = Km*Kg/(R*r)
K2 = Km**2*Kg**2/(R*r**2)
l = 1.0  # length of pendulum to CG
#t1 = 1.0
t2 = 0.0
I = 0.006  # inertia of the pendulum                


L = (I + m*l**2)/(m*l)
g = 9.81  # gravity
Vsat = 20.  # saturation voltage

'''A11 = -1 * Km**2*Kg**2 / ((M - m*l/L)*R*r**2)
A12 = -1*g*m*l / (L*(M - m*l/L))
A31 = Km**2*Kg**2 / (M*(L - m*l/M)*R*r**2)
A32 = g/(L-m*l/M)'''
A11 = (8./3.)*m*l**2
A12 = (1./3.)*m*l**2
A31 = (1./3.)*m*l**2
A32 = (2./3.)*m*l**2
A = matrix([
    [0, 1, 0, 0],
    [0, A11, A12, 0],
    [0, 0, 0, 1],
    [0, A31, A32, 0]
])

B1 = 3.2
B2 = 1.8

B = matrix([
    [0],     
    [B1],
    [0],
    [B2]
])
Q = matrix([
    [1000, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1000, 0],
    [0, 0, 0, 1]
])

(KT, S, E) = lqr(A, B, Q, 2.0);
K = place(A, B, [-0.223, 0.0, -1.0, 0.0])

def constrain(theta):
    theta = theta % (2*pi)
    if theta > pi:
        theta = -2*pi+theta
    return theta

def sat(Vsat, V):
    if abs(V) > Vsat:
        return Vsat * cmp(V, 0)
    return V

def average(x):
    x_i, k1, k2, k3, k4 = x
    return x_i + (k1 + 2.0*(k3 + k4) +  k2) / 6.0

theta = []
class Pendulum(object):
    def __init__(self, dt, init_conds, end):
        self.dt = dt
        self.t = 0.0
        self.x = init_conds[:]
        self.end = end

    def derivative(self, u):
        V = sat(Vsat, self.control(u))
        #x1 = x, x2 = x_dt, x3 = theta, x4 = theta_dt
        x1, y1, x2, y2 = u
        x1_dt, x2_dt =  y1, y2
        y1_dt = ((2./3.)*((V/(0.5*m*l**2)+(np.sin(x2)*y2*(2*y1+y2))))-(((2./3.)+(np.cos(x2))*((t2/0.5*m*l**2))-np.sin(x2)*y1**2)))/((16./9.)-(np.cos(x2))**2)
        y2_dt = -(((2./3.)+np.cos(x2))*((V/(0.5*m*l**2))+np.sin(x2)*y2*(2*y1+y2))+2*(((5./3.)+np.cos(x2))*(t2/0.5*m*l**2)-np.sin(x2)*y1**2))/((16./9.)-(np.cos(x2))**2)
        x = [x1_dt, y1_dt, x2_dt, y2_dt]
        return x
        
    def control(self, u):
        c = constrain(u[2])
        if c>-pi/5.0 and c<pi/5.0:
            return float(-K*matrix(u[0:2]+[c]+[u[3]]).T)
        else:
            return self.swing_up(u)

    def swing_up(self, u):
        #E1 = 0.5*0.67*m*l**2*u[1]**2
        E0 = 0.0
        k = 1
        w = 2*(m*g*l/(4*I))**(.5)
        E = m*g*l*(.5*(u[1]/w)**2 + cos(u[0])-1)
        a = k*(E-E0)*cmp(u[3]*cos(u[2]), 0)
        F = m*a
        V = (F - K2*constrain(u[2]))/K1
        #pend.input['thetas'] = u[2]
        #pend.input['omegas'] = u[3]
        #pend.compute()
        #V = pend.output['force']
        return sat(Vsat, V)

    def rk4_step(self, dt):
        dx = self.derivative(self.x)
        k2 = [ dx_i*dt for dx_i in dx ]

        xv = [x_i + delx0_i/2.0 for x_i, delx0_i in zip(self.x, k2)]
        k3 = [ dx_i*dt for dx_i in self.derivative(xv)]

        xv = [x_i + delx1_i/2.0 for x_i,delx1_i in zip(self.x, k3)]
        k4 = [ dx_i*dt for dx_i in self.derivative(xv) ]

        xv = [x_i + delx1_2 for x_i,delx1_2 in zip(self.x, k4)]
        k1 = [self.dt*i for i in self.derivative(xv)]

        self.t += dt
        self.x = map(average, zip(self.x, k1, k2, k3, k4))
        theta.append(constrain(self.x[2]))


    def integrate(self):
        x = []
        while self.t <= self.end:
            self.rk4_step(self.dt)
            x.append([self.t] + self.x)
        return array(x)