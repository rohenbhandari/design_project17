import control
from control import *
from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
M11, M12, M21, M22 = symbols('M11 M12 M21 M22')
m1, m2, l1, l2 = symbols('m1 m2 l1 l2')
M11 = (0.33*m1 + m2)*l1**2 + 0.33*m2*l2**2 + m2*l2**2
M11 = M11.subs([(m1, 1), (m2, 1), (l1, 1), (l2, 1)])
M12 = (0.33 * m2 * l2**2) + (m1 * l1) + (m2 * l2)
M12 = M12.subs([(m1, 1), (l1, 1), (m2, 1), (l2, 1)])
M21 = M12.subs([(m1, 1), (l1, 1), (m2, 1), (l2, 1)])
M22 = (0.33 * m2 * l2**2)
M22 = M22.subs([(m2, 1), (l2, 1)])
sys1 = control.ss([[1.84, 2.36], [2.3, 0.3]], "5.34; 7.56", "6. 8", "9.")
print sys1
E = np.matrix([[2.6, 2.36], [3.11, 4.56]])
F = np.matrix("5.; 7")
#print control.feedback(sys1, sys2=1, sign=-1)
#print control.acker(E, F, [0, 1])
#print control.ctrb(E, F)
#print control.dcgain(sys1)
#print control.root_locus(sys1)
#print control.ss2tf(sys1)
#mag, phase, omega = control.bode(sys1)
#plt.show()
#real, imag, freq = control.nyquist_plot(sys1)
#plt.show()
#plt.plot(control.initial_response(sys1))
#plt.show()
#K = control.place(E, F, [-2, -5])
#print K