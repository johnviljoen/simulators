"""
File containing trimming scripts for 4 states of flight:

1. Steady wings-level
2. Steady turning
3. Steady pull-up
4. Steady roll

An example of how to use the code is contained in the __main__ section below.

I have validated each of these with the equivalent MATLAB functions
using fminsearch, which uses Nelder-Mead, like the scipy.optimize.fmin
that I have used here.

I have quite a lot of duplicate code here for verbosity and to make sure
that the user must explicitly decide the type of trimming that is conducted.

I could definitely have made this code simpler. It could be one trim function
that makes no assumptions about the type of manouvre for example, you would
just have to specify the phi, theta, psi weightings and the desired p,q,r.
"""

from scipy import optimize
import numpy as np
import common.pytorch_utils as ptu
from f16.parameters import u_ub, u_lb, x_ub, x_lb
import pickle

def obj_func(UX0, h_t, v_t, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant):

    """
    {UX0, h_t, v_t,  phi, psi, p,     q,     r,     nlplant}
    {mix, ft,  ft/s, deg, deg, deg/s, deg/s ,deg/s, NLplant}
    """

    # calculate dLEF
    rho0 = 2.377e-3
    tfac = 1 - 0.703e-5*h_t
    temp = 519*tfac
    if h_t >= 35000:
        temp = 390
    rho = rho0*tfac**4.14
    qbar = 0.5*rho*v_t**2
    ps = 1715*rho*temp
    dLEF = 1.38*UX0[2]*180/np.pi - 9.05*qbar/ps + 1.45

    # for xu
    xu = np.array([
        0,                  # npos
        0,                  # epos
        h_t,                # altitude (ft)
        phi*np.pi/180,      # phi (rad)
        UX0[2],             # theta (rad)
        psi*np.pi/180,      # psi (rad)
        v_t,                # velocity (ft/s)
        UX0[2],             # alpha (rad)
        0,                  # beta (rad)
        p*np.pi/180,        # p (rad/s)
        q*np.pi/180,        # q ( rad/s)
        r*np.pi/180,        # r (rad/s)
        UX0[0],             # thrust (lbs)
        UX0[1],             # elevator (deg)
        UX0[3],             # aileron (deg)
        UX0[4],             # rudder (deg)
        dLEF,               # dLEF (deg)
        -UX0[2]*180/np.pi   # LF_state (deg)
    ])
    
    # thrust limits
    xu[12] = np.clip(xu[12], a_min=u_lb[0], a_max=u_ub[0])
    # elevator limits
    xu[13] = np.clip(xu[13], a_min=u_lb[1], a_max=u_ub[1])
    # aileron limits
    xu[14] = np.clip(xu[14], a_min=u_lb[2], a_max=u_ub[2])
    # rudder limits
    xu[15] = np.clip(xu[15], a_min=u_lb[3], a_max=u_ub[3])
    # alpha limits
    xu[7] = np.clip(xu[7], a_min=x_lb[7]*np.pi/180, a_max=x_ub[7]*np.pi/180)

    u = xu[12:16]
    xdot = ptu.to_numpy(nlplant.forward(ptu.from_numpy(xu), ptu.from_numpy(u))[0])
    xdot = xdot.reshape([18,1])

    

    weight = np.array([0, 0, 5, phi_w, theta_w, psi_w, 2, 10, 10, 10, 10, 10]).reshape([1,12])
    cost = weight @ (xdot[0:12]**2)
    return cost

def wings_level(thrust, elevator, alpha, ail, rud, vel, alt, nlplant, verbose=False):

    """
    provide initial guesses for thrust, elevator, alpha, aileron, and rudder
    for a given velocity and altitude for steady wings level flight.

    {thrust, elevator, alpha, aileron, rudder, velocity, altitude, nlplant}
    {lbs,    deg,      deg,   deg,     deg,    ft/s,     ft,       NLPlant}
    """

    NUMBER_OF_NM_OPTIMISATIONS = 2

    # initialise euler angle weightings for the objective function
    phi_w = 10
    theta_w = 10
    psi_w = 10

    # initialise the other required variables to zero
    phi, psi = 0, 0
    p, q, r = [0]*3

    UX0 = np.array([
        thrust,
        elevator,
        alpha*np.pi/180,
        rud,
        ail
    ])

    UX = UX0 # deepcopy this if you want to analyse optimisation

    for i in range(NUMBER_OF_NM_OPTIMISATIONS):
        UX = optimize.fmin(obj_func, UX, args=((alt, vel, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant)), xtol=1e-10, ftol=1e-10, maxfun=5e+04, maxiter=1e+04, disp=verbose)

    return UX

def turning(thrust, elevator, alpha, ail, rud, vel, alt, turning_rate, nlplant, verbose=False):

    """
    provide initial guesses for thrust, elevator, alpha, aileron, and rudder
    for a given velocity and altitude for steady turning flight.

    {thrust, elevator, alpha, aileron, rudder, velocity, altitude, turning_rate, nlplant}
    {lbs,    deg,      deg,   deg,     deg,    ft/s,     ft,       deg/s,        NLPlant}
    """

    NUMBER_OF_NM_OPTIMISATIONS = 2

    # initialise euler angle weightings for the objective function
    phi_w = 10
    theta_w = 10

    # initialise the other required variables to zero
    phi, psi = 0, 0
    p, q = 0, 0
    r = turning_rate
    psi_w = 0

    UX0 = np.array([
        thrust,
        elevator,
        alpha*np.pi/180,
        rud,
        ail
    ])

    UX = UX0

    for i in range(NUMBER_OF_NM_OPTIMISATIONS):
        UX = optimize.fmin(obj_func, UX, args=((alt, vel, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant)), xtol=1e-10, ftol=1e-10, maxfun=5e+04, maxiter=1e+04, disp=verbose)

    return UX

def pull_up(thrust, elevator, alpha, ail, rud, vel, alt, pull_up_rate, nlplant, verbose=False):

    """
    provide initial guesses for thrust, elevator, alpha, aileron, and rudder
    for a given velocity and altitude for steady pull up flight.

    {thrust, elevator, alpha, aileron, rudder, velocity, altitude, pull_up_rate, nlplant}
    {lbs,    deg,      deg,   deg,     deg,    ft/s,     ft,       deg/s,        NLPlant}
    """

    NUMBER_OF_NM_OPTIMISATIONS = 2

    # initialise euler angle weightings for the objective function
    phi_w = 10
    theta_w = 0
    psi_w = 10

    # initialise the other required variables to zero
    phi, psi = 0, 0
    p, r = 0, 0
    q = pull_up_rate

    UX0 = np.array([
        thrust,
        elevator,
        alpha*np.pi/180,
        rud,
        ail
    ])

    UX = UX0

    for i in range(NUMBER_OF_NM_OPTIMISATIONS):
        UX = optimize.fmin(obj_func, UX, args=((alt, vel, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant)), xtol=1e-10, ftol=1e-10, maxfun=5e+04, maxiter=1e+04, disp=verbose)

    return UX

def roll(thrust, elevator, alpha, ail, rud, vel, alt, roll_rate, nlplant, verbose=False):

    """
    provide initial guesses for thrust, elevator, alpha, aileron, and rudder
    for a given velocity and altitude for steady pull up flight.

    {thrust, elevator, alpha, aileron, rudder, velocity, altitude, roll_rate, nlplant}
    {lbs,    deg,      deg,   deg,     deg,    ft/s,     ft,       deg/s,     NLPlant}
    """

    NUMBER_OF_NM_OPTIMISATIONS = 2

    # initialise euler angle weightings for the objective function
    phi_w = 0
    theta_w = 10
    psi_w = 10

    # initialise the other required variables to zero
    phi, psi = 0, 0
    q, r = 0, 0
    p = roll_rate

    UX0 = np.array([
        thrust,
        elevator,
        alpha*np.pi/180,
        rud,
        ail
    ])

    UX = UX0

    for i in range(NUMBER_OF_NM_OPTIMISATIONS):
        UX = optimize.fmin(obj_func, UX, args=((alt, vel, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant)), xtol=1e-10, ftol=1e-10, maxfun=5e+04, maxiter=1e+04, disp=verbose)

    return UX

if __name__ == "__main__":

    """
    An example of how you can use the code in this file
    """

    import dynamics
    from copy import deepcopy

    ALLOWABLE_OBJ_FUNC_ERROR = 1e-03
    DIRECTORY_PATH = 'tests/MATLAB_timehistory/'
    NUMBER_OF_NM_OPTIMISATIONS = 2

    h_t = 10000 # ft
    v_t = 700 # ft/s

    nlplant = dynamics.NLPlant()

    # the MATLAB timeshistories contain trimmed initial states to compare against
    def load_pkl(dir_path, filename):
        # open the file
        file_to_read = open(dir_path + filename + '.pkl', "rb")
        # load the pickle
        data = pickle.load(file_to_read)
        file_to_read.close()
        return data

    filename = '10000ft_700fts_x_sim'
    trim_ref = load_pkl(DIRECTORY_PATH, filename)['x_sim'][0,:]

    # initial guess
    thrust = 5000;          # thrust, lbs
    elevator = -0.09;       # elevator, degrees
    alpha = 8.49;           # AOA, degrees
    rudder = 0.01;         # rudder angle, degrees
    aileron = -0.01;         # aileron, degrees

    # initialise the other required variables to zero
    phi, psi, p, q, r = [0]*5

    UX0 = np.array([
        thrust,
        elevator,
        alpha*np.pi/180,
        rudder,
        aileron
    ])

    phi_w = 10
    theta_w = 10
    psi_w = 10


    # assert cost == 150.6866 # same as MATLAB

    # check objective function outputs
    cost_init = obj_func(UX0, h_t, v_t, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant)

    assert np.abs(cost_init - 150.6866) < ALLOWABLE_OBJ_FUNC_ERROR

    UX = deepcopy(UX0)

    for i in range(NUMBER_OF_NM_OPTIMISATIONS):
        UX = optimize.fmin(obj_func, UX, args=((h_t, v_t, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant)), xtol=1e-10, ftol=1e-10, maxfun=5e+04, maxiter=1e+04, disp=False)

    cost_final = obj_func(UX, h_t, v_t, phi, psi, p, q, r, phi_w, theta_w, psi_w, nlplant)

    print(cost_final)

    test_steady = wings_level(thrust, elevator, alpha, aileron, rudder, v_t, h_t, nlplant, verbose=True)

    turning_rate = 3 #deg/s
    test_turn = turning(thrust, elevator, alpha, aileron, rudder, v_t, h_t, turning_rate, nlplant, verbose=True)

    pull_up_rate = 2 #deg/s
    test_pullup = pull_up(thrust, elevator, alpha, aileron, rudder, v_t, h_t, pull_up_rate, nlplant, verbose=True)

    roll_rate = 5 #deg/s
    test_roll = roll(thrust, elevator, alpha, aileron, rudder, v_t, h_t, roll_rate, nlplant, verbose=True)

    print(test_pullup)

