from matplotlib import pyplot as plt 
from numpy import linspace, sin

def func(t, A, B, C):
    return t*sin(A*t) + B * C*t

tspan = linspace(0, 10, 101)
plt.plot(tspan, func(tspan, 2,3,5))
plt.show()