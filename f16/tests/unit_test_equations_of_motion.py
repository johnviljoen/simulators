"""
3 things to compare:
- MATLAB saved output
- nlplant.so file
- Python LUT implementation

Lets find out if the nlplant.so is matching with the MATLAB, indicating
a discrepancy between its dynamics and the python EoM

The result was that nlplant.so agreed with the Python, and so MATLAB seems
to be wrong.
"""
import dynamics
import pytorch_utils as ptu
import numpy as np
from ctypes import CDLL
import ctypes
import os

ALLOWABLE_ERROR = 1e-05

d2r = np.pi/180

# MATLAB reference first two states using dt = 0.1

x0 = np.array([ 0.0000000e+00,  0.0000000e+00,  1.0000000e+04,  0.0000000e+00,
        1.0721306e+00*np.pi/180,  0.0000000e+00,  7.0000000e+02,  1.0721306e+00*np.pi/180,
        0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        2.7574448e+03, -1.2709435e+00, -8.9479364e-02, -4.0362250e-02,
       -1.0721306e+00*np.pi/180,  0.0000000e+00])
u0 = np.array([ 2.7574e+03, -1.2709e+00, -8.9479e-02, -4.0362e-02]) 
x1 = np.array([ 7.0000e+01, -1.4904e-03,  1.0000e+04,  1.3847e-04*d2r,  1.0721e+00*d2r,
        -4.6765e-05*d2r,  7.0000e+02,  1.0722e+00*d2r, -2.3781e-03*d2r,  4.0189e-03*d2r,
        -1.2092e-05*d2r, -1.3769e-03*d2r,  2.7574e+03, -1.2709e+00, -8.9479e-02,
        -4.0362e-02, -1.0722e+00*d2r,  0.0000e+00]) 
u1 = np.array([ 2.7574e+03, -1.2709e+00, -8.9479e-02, -4.0362e-02]) 

# make an nlplant instance
nlplant = dynamics.NLPlant()

# make an independant copy of the initial conditions
x0_sim = np.copy(x0)
u0_sim = np.copy(u0)
x0_ori = np.copy(x0)
u0_ori = np.copy(u0)

# find the equations_of_motion xdot output
xdot_py_eom = nlplant.equations_of_motion(ptu.from_numpy(x0_sim))[0]

# retrieve the reference xdot found from calling nlplant in script in MATLAB
xdot_matlab = np.array([700,
0,
0,
0,
0,
0,
0.0477741519519715,
-0.000218861186507004,
-0.000419839889432380,
2.75450475584100e-05,
0.00854796544248329,
-0.000499424249688084])



# setup nlplant.so
so_file = os.getcwd() + "/aerodata/nlplant_reference.so"
nlplant_so = CDLL(so_file)

# initialise xu and xdot
xdot_c_eom = np.zeros(18)
fi_flag = 1

nlplant_so.nlplant(ctypes.c_void_p(x0_ori.ctypes.data), ctypes.c_void_p(xdot_c_eom.ctypes.data))

# test the original nlplant.c against the dynamics.NLPlant.equations_of_motion reimplementation:
assert np.abs(xdot_c_eom[:12] - xdot_py_eom.numpy()[:12]).max() < ALLOWABLE_ERROR
print('original nlplant.c is within numerical error of dynamics.NLPlant.equations_of_motion')

# test the original MATLAB xdot with the Python to ensure independent correctness
assert np.abs(xdot_c_eom[:12] - xdot_matlab).max() < ALLOWABLE_ERROR
print('original matlab xdot calculation within numerical error of nlplant.c output in Python')

print('verified that the MATLAB sim == Python C sim == Python Py sim')

print(np.abs(xdot_c_eom[:12] - xdot_matlab).max())



# ok so the xdots are within e-18 of each other... then why does the end to end sim diverge?




