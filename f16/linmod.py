"""
This file contains a numerical and analytical lineariser for the F16 dynamics

examples of how to use all parts are shown in the __main__ section below.
"""

import numpy as np
import torch
import functorch
import dynamics
import common.pytorch_utils as ptu
import torch

nlplant = dynamics.NLPlant(lookup_type='Py') # lookup_type is Py for torch jacobian

# Copied from the python control systems library
# Source: https://github.com/python-control/python-control/blob/main/control/iosys.py
def linmod_numerical(nlplant, x, u, eps=1e-3):

    """
    This function works poorly when different states are of
    different orders of magnitude. For optimal results input output
    should be normalised before and after.

    This is not so easy with the nonlinear plant which has been
    built, and so an analytical differentiation is preferable using
    torch functional jacobians.

    Furthermore the conversion from numpy to torch during the loop
    is rather inefficient as it stands imo.
    """

    # n = number of states
    # m = number of inputs
    # o = number of outputs
    n = len(x)
    m = len(u)
    o = len(nlplant.obs(x,u))

    # Compute the nominal value of the update law and output
    F0 = nlplant.forward(x, u)[0]
    H0 = nlplant.obs(x, u)[0]
    
    A = ptu.from_numpy(np.zeros([n,n]))
    B = ptu.from_numpy(np.zeros([n,m]))
    C = ptu.from_numpy(np.zeros([o,n]))
    D = ptu.from_numpy(np.zeros([o,m]))

    # Perturb each of the state variables and compute linearization
    for i in range(n):
        dx = ptu.from_numpy(np.zeros((n,)))
        dx[i] = eps
        A[:, i] = (nlplant.forward(x + dx, u)[0] - F0) / eps
        C[:, i] = (nlplant.obs(x + dx, u) - H0) / eps

    # Perturb each of the input variables and compute linearization
    for i in range(m):
        du = ptu.from_numpy(np.zeros((m,)))
        du[i] = eps
        B[:, i] = (nlplant.forward(x, u + du)[0] - F0) / eps
        D[:, i] = (nlplant.obs(x, u + du) - H0) / eps
    
    return A, B, C, D


def xdot_nlplant_wrapper(x,u):
    return nlplant.forward(x,u)[0]

def obs_nlplant_wrapper(x,u):
    return nlplant.obs(x,u)

def linmod_jacobian(nlplant, x0, u0, differentiable=False):
    """
    This computes the linearisation with the jacobian torch function, which
    envokes the autograd.grad once per row of the jacobian. This is slower than
    if we used vectorized tools like jacrev and jacfwd

    This function can however be made differentiable, unlike jacrev and jacfwd
    from my understanding.
    """
    A, B = torch.autograd.functional.jacobian(xdot_nlplant_wrapper, (x0,u0), create_graph=differentiable)
    C, D = torch.autograd.functional.jacobian(obs_nlplant_wrapper, (x0,u0), create_graph=differentiable)
    return A, B, C, D

def linmod_jacrev():
    """
    Does not work with lookup table implementations as they do not have
    differentiability at every point
    """
    A_func = functorch.jacfwd(xdot_nlplant_wrapper, argnums=0)
    B_func = functorch.jacfwd(xdot_nlplant_wrapper, argnums=1)
    C_func = functorch.jacfwd(obs_nlplant_wrapper, argnums=0)
    D_func = functorch.jacfwd(obs_nlplant_wrapper, argnums=1)

    return A_func, B_func, C_func, D_func
    


if __name__ == "__main__":

    import trim
    import copy

    # get an initial condition by trimming the aircraft at wings level flight
    # using a couple initial guesses

    # start at velocity of 700 ft/s at 10000 ft altitude
    altitude = 10000 # ft
    velocity = 700   # ft/s

    thrust = 5000;          # thrust, lbs
    elevator = -0.09;       # elevator, degrees
    alpha = 8.49;           # AOA, degrees
    rudder = 0.01;          # rudder angle, degrees
    aileron = -0.01;        # aileron, degrees

    trim_states = trim.wings_level(thrust, elevator, alpha, aileron, rudder, velocity, altitude, nlplant, verbose=True)

    # calculate dLEF
    rho0 = 2.377e-3
    tfac = 1 - 0.703e-5*altitude
    temp = 519*tfac
    if altitude >= 35000:
        temp = 390
    rho = rho0*tfac**4.14
    qbar = 0.5*rho*velocity**2
    ps = 1715*rho*temp
    dLEF = 1.38*trim_states[2]*180/np.pi - 9.05*qbar/ps + 1.45

    x0_np = np.array([
            0,                  # npos
            0,                  # epos
            altitude,                # altitude (ft)
            0*np.pi/180,      # phi (rad)
            trim_states[2],             # theta (rad)
            0*np.pi/180,      # psi (rad)
            velocity,                # velocity (ft/s)
            trim_states[2],             # alpha (rad)
            0,                  # beta (rad)
            0*np.pi/180,        # p (rad/s)
            0*np.pi/180,        # q ( rad/s)
            0*np.pi/180,        # r (rad/s)
            trim_states[0],             # thrust (lbs)
            trim_states[1],             # elevator (deg)
            trim_states[3],             # aileron (deg)
            trim_states[4],             # rudder (deg)
            dLEF,               # dLEF (deg)
            -trim_states[2]*180/np.pi   # LF_state (deg)
        ])

    x0 = ptu.from_numpy(copy.deepcopy(x0_np))
    u0 = ptu.from_numpy(copy.deepcopy(x0_np[12:16]))

    A,B,C,D = linmod_numerical(nlplant, x0, u0, eps=1e-2)

    torch.autograd.set_detect_anomaly(True)

    At, Bt, Ct, Dt = linmod_jacobian(nlplant, x0, u0, differentiable=False)

    # my word this actually works
    A_func, B_func, C_func, D_func = linmod_jacrev()
    Ajrev = A_func(x0, u0)
    Bjrev = B_func(x0, u0)
    Cjrev = C_func(x0, u0)
    Djrev = D_func(x0, u0)

    # the non stationary offset is the nonlinear xdot at a particular state
    offset = nlplant.forward(x0, u0)[0]

    # linmod_jacrev(nlplant, x0, u0)
    print(At)
