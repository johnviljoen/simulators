"""
Putting this file here so that I may include complex time
dependent integrators in the future, but for now its just 
Euler and RK4
"""

def Euler(f, x, u, Ts, *args):
    return x + Ts * f(x, u, *args)

def RK4(f, x, u, Ts, *args):
    k1 = f(x, u, *args)
    k2 = f(x + Ts / 2 * k1, u, *args)
    k3 = f(x + Ts / 2 * k2, u, *args)
    k4 = f(x + Ts * k3, u, *args)
    return x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)