"""
This script contains a lot of the testing I carried out whilst originally
writing the code, it is not designed to be run. I have left it here
as a record of much of my thought process when debugging. It is possible
segments of it can be used to understand how to interact with different
parts of the code at a low level, however I have not built any guide 
to do so. Therefore feel free to get confused at my spaghetti at your
own risk.
"""

import pytorch_utils as ptu
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# testing gpu activation
ptu.init_gpu(
    use_gpu=not False,
    gpu_id=0
)

# test instantiation of NLPlant
import dynamics

nlplant = dynamics.NLPlant(lookup_type='C')

# test nlplant forward function
import params
x0 = ptu.from_numpy(params.x0)
u0 = ptu.from_numpy(params.u0)

xdot, accels, atmos = nlplant.forward(x0, u0)

# test python lookup tables for simulation using just Cx
from tables import PyLookupTorch, CLookup

py_lookup = PyLookupTorch()
c_lookup = CLookup()

point = np.array([-2.5, 1.0, 1.0])
py_lookup.get_bounds_3d(np.array([0.,0.1,0.1]), 'Cx')

py_lookup.get_bounds_2d(np.array([0.,0.1]), 'Cx_lef')
py_lookup.get_bounds_1d(np.array([0.]), 'CXq')

CXq = py_lookup.interp_1d(np.array([0.]), 'CXq')
Cx_lef = py_lookup.interp_2d(np.array([0.,0.]), 'Cx_lef')

Cx_py = py_lookup.interp_3d(point, 'Cx')

Cx, _, _, _, _, _ = c_lookup.hifi_C(point)

diff_Cx = Cx - Cx_py

# test nlplant with lookup_type = 'Py'

nlplant_py = dynamics.NLPlant(lookup_type='Py')

xdot_py, accels_py, atmos_py = nlplant_py.forward(x0, u0)

# we want this to be zero
xdot_diff = xdot - xdot_py

# test an end to end simulation on both the C and the Python

def autonomous_sim(nlplant, x0, u, t_end, dt):
    print(f"Using {nlplant.lookup_type} LUTs")
    x_traj = ptu.from_numpy(np.zeros([int(t_end/dt),len(x0)]))
    x = torch.clone(x0)
    for idx, t in tqdm(enumerate(range(0,int(t_end/dt),t_end))):
        xdot = nlplant.forward(x0, u)[0]
        x += xdot*dt
        x_traj[idx,:] = x 
    return x_traj

c_x_traj = autonomous_sim(nlplant, x0, u0, 1, 1e-2)
py_x_traj = autonomous_sim(nlplant_py, x0, u0, 1, 1e-2)

diff = (c_x_traj - py_x_traj).abs().max()

# test out of bounds behaviour between Py and C lookups
# x_out_of_bounds = x_oob
x_oob = torch.clone(x0)
x_oob[7] = 100. # sets alpha to 100 > 90

# Result: C seg faults, Py finds out of bounds in interp3d line 330
#c_x_traj = autonomous_sim(nlplant, x_oob, u0, 1, 1e-2)
#py_x_traj = autonomous_sim(nlplant_py, x_oob, u0, 1, 1e-2)

# test deep alpha regime, some LUTs operate up to 45, others to 90 for alpha
x_dar = torch.clone(x0)
x_dar[7] = 90. * torch.pi/180

c_x_traj = autonomous_sim(nlplant, x_dar, u0, 1, 1e-2)
py_x_traj = autonomous_sim(nlplant_py, x_dar, u0, 1, 1e-2)

diff_dar = (c_x_traj - py_x_traj).abs().max()
diff_dar0 = (c_x_traj - py_x_traj)[0,:].abs().max()

# plt.plot(py_x_traj[:,7].detach().cpu().numpy())
# plt.plot(c_x_traj[:,7].detach().cpu().numpy())
# plt.show()

# test extrapolation of the tables.py interp1d,2d methods individually

"""
HIGHEST ACCEPTABLE diff_dar0 SHOULD BE 1e-04 due to tables being 4 SF at minimum, accumulated
errors are acceptable
"""

assert diff_dar0 < 1e-4

# for beta
x_dar = torch.clone(x0)
x_dar[8] = 29. * torch.pi/180

c_x_traj = autonomous_sim(nlplant, x_dar, u0, 1, 1e-2)
py_x_traj = autonomous_sim(nlplant_py, x_dar, u0, 1, 1e-2)

diff_dar = (c_x_traj - py_x_traj).abs().max()
diff_dar0 = (c_x_traj - py_x_traj)[0,:].abs().max()

# plt.plot(py_x_traj[:,7].detach().cpu().numpy())
# plt.plot(c_x_traj[:,7].detach().cpu().numpy())
# plt.show()

assert diff_dar0 < 1e-4

"""
Debugging the 3d tables
"""

"""
I have found that the 3D tables are incorrect currently. (This has been fixed)

I have a suspicion that the dh axis should just be flipped (This was not the problem)
"""

import dynamics
import params
from params import c2f
import pytorch_utils as ptu
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

nlplant = dynamics.NLPlant()

# lets take a cross section of inp = [-20,-30,i], where i is dh sweep -25 -10 0 10 25

diff_Cn_1 = []
diff_Cl_1 = []
beta = -30.
dh = 0.
for i, alpha in enumerate(nlplant.py_lookup.axes['ALPHA1']):
    inp = np.array([alpha,beta,dh])
    temp = np.concatenate([inp[0:2], np.array([0])])
    c_Cx, c_Cz, c_Cm, c_Cy, c_Cn, c_Cl = nlplant.c_lookup.hifi_C(inp)
    py_Cy = nlplant.py_lookup.interp_2d(inp[0:2], 'Cy')
    py_Cn = nlplant.py_lookup.interp_3d(inp, 'Cn')
    py_Cl = nlplant.py_lookup.interp_3d(inp, 'Cl')

    diff_Cn_1.append(py_Cn - c_Cn)
    diff_Cl_1.append(py_Cl - c_Cl)


diff_Cn_2 = []
diff_Cl_2 = []
beta = 30.
dh = 0.
for i, alpha in enumerate(nlplant.py_lookup.axes['ALPHA1']):
    inp = np.array([alpha,beta,dh])
    temp = np.concatenate([inp[0:2], np.array([0])])
    c_Cx, c_Cz, c_Cm, c_Cy, c_Cn, c_Cl = nlplant.c_lookup.hifi_C(inp)
    py_Cy = nlplant.py_lookup.interp_2d(inp[0:2], 'Cy')
    py_Cn = nlplant.py_lookup.interp_3d(inp, 'Cn')
    py_Cl = nlplant.py_lookup.interp_3d(inp, 'Cl')

    diff_Cn_2.append(py_Cn - c_Cn)
    diff_Cl_2.append(py_Cl - c_Cl)

assert max(diff_Cl_1) < 1e-04
assert max(diff_Cl_2) < 1e-04
assert max(diff_Cn_1) < 1e-04
assert max(diff_Cn_2) < 1e-04


# plt.plot(diff_Cn_1)
# plt.plot(diff_Cn_2)
# plt.plot(diff_Cl_1)
# plt.plot(diff_Cl_2)
# plt.show()

"""
Debugging the rudder tables
"""

"""
I am currently encountering a bug in the rudder tables during integration
testing in integration_test_tables.py, and nowhere else.

The structure of these lookups in the C are as follows:

void hifi_rudder(double alpha, double beta, double *retVal){
        retVal[0] = _Cy_r30(alpha,beta) - _Cy(alpha,beta);
        retVal[1] = _Cn_r30(alpha,beta) - _Cn(alpha,beta,0);
        retVal[2] = _Cl_r30(alpha,beta) - _Cl(alpha,beta,0);
}

None of these tables are clipped, all are full fidelity. 

I know Cy,n,l to be correct due to them not showing any deviation
in the integration tests.

Therefore I plan to isolate the _Cy,n,l_r30 by adding to the
retVal Cy,n,l which I know to be correct respectively.

SOLVED: The problem was I was not clipping the input to the rud tables
as the C appears to be doing, despite the tables being able to not be
clipped. I am confused why the C is written this way, but it could be
that they found out the high fidelity tables for these coefficients
were inaccurate.
"""

import dynamics
import params
from params import c2f
import pytorch_utils as ptu
import numpy as np
import torch
from tqdm import tqdm

nlplant = dynamics.NLPlant()

diff_max_delta_Cy_30 = ptu.from_numpy(np.array(0.))
diff_max_delta_Cn_30 = ptu.from_numpy(np.array(0.))
diff_max_delta_Cl_30 = ptu.from_numpy(np.array(0.))
for i, alpha in tqdm(enumerate(nlplant.py_lookup.axes['ALPHA1'])):
    for j, beta in enumerate(nlplant.py_lookup.axes['BETA1']):
        for k, dh in enumerate(nlplant.py_lookup.axes['DH1']):
            inp = np.array([alpha,beta,dh])
            temp = np.concatenate([inp[0:2], np.array([0])])

            py_Cy = nlplant.py_lookup.interp_2d(inp[0:2], 'Cy')
            py_Cn = nlplant.py_lookup.interp_3d(temp, 'Cn')
            py_Cl = nlplant.py_lookup.interp_3d(temp, 'Cl') 
            py_Cy_r30 = nlplant.py_lookup.interp_2d(inp[0:2], 'Cy_r30')
            py_Cn_r30 = nlplant.py_lookup.interp_2d(inp[0:2], 'Cn_r30')
            py_Cl_r30 = nlplant.py_lookup.interp_2d(inp[0:2], 'Cl_r30')

            c_delta_Cy_r30, c_delta_Cn_r30, c_delta_Cl_r30 = nlplant.c_lookup.hifi_rudder(temp)
            c_Cx, c_Cz, c_Cm, c_Cy, c_Cn, c_Cl = nlplant.c_lookup.hifi_C(temp)
            c_Cy_r30 = c_delta_Cy_r30 + c_Cy
            c_Cn_r30 = c_delta_Cn_r30 + c_Cn
            c_Cl_r30 = c_delta_Cl_r30 + c_Cl

            Cn_diff = c_Cn - py_Cn
            Cl_diff = c_Cl - py_Cl
            Cn_r30_diff = c_Cn_r30 - py_Cn_r30
            Cl_r30_diff = c_Cl_r30 - py_Cl_r30

            # the below showed that the difference is entirely due to the differences in the 3D tables
            # print(f'Cn_diff: {Cn_diff}')
            # print(f'Cn_r30_diff: {Cn_r30_diff}')
            # print(f'Cl_diff: {Cl_diff}')
            # print(f'Cl_r30_diff: {Cl_r30_diff}')

            py_delta_Cy_r30 = nlplant.py_lookup.interp_2d(inp[0:2], 'Cy_r30') - nlplant.py_lookup.interp_2d(inp[0:2], 'Cy')
            py_delta_Cn_r30 = nlplant.py_lookup.interp_2d(inp[0:2], 'Cn_r30') - nlplant.py_lookup.interp_3d(temp, 'Cn')
            py_delta_Cl_r30 = nlplant.py_lookup.interp_2d(inp[0:2], 'Cl_r30') - nlplant.py_lookup.interp_3d(temp, 'Cl')

            diff_max_delta_Cy_30 = torch.max(diff_max_delta_Cy_30, py_delta_Cy_r30 - c_delta_Cy_r30) 
            diff_max_delta_Cn_30 = torch.max(diff_max_delta_Cn_30, py_delta_Cn_r30 - c_delta_Cn_r30) 
            diff_max_delta_Cl_30 = torch.max(diff_max_delta_Cl_30, py_delta_Cl_r30 - c_delta_Cl_r30) 

            # print(f'diff_delta_Cy_r30: {py_delta_Cy_r30 - c_delta_Cy_r30}')
            # print(f'diff_delta_Cn_r30: {py_delta_Cn_r30 - c_delta_Cn_r30}')
            # print(f'diff_delta_Cl_r30: {py_delta_Cl_r30 - c_delta_Cl_r30}')

assert diff_max_delta_Cy_30 < 1e-4
assert diff_max_delta_Cn_30 < 1e-4
assert diff_max_delta_Cl_30 < 1e-4


