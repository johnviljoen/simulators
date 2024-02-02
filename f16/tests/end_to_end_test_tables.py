"""
This sanity check end-to-end test makes sure the largest discrepancy over 1000 iterations of
the simulation is no more than 1e-04. Sure this is a limited check as some states are much 
larger than others, so this is more than likely just showing that the velocity has not diverged
haha, but its a quick sanity check of the whole simulation end to end.

It verifies that the C lookup simulation and the Python lookup simulation are the same within
a numerical tolerance.
"""

import dynamics
import pytorch_utils as ptu
import params
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

print('Performing end to end autonomous simulation test to compare C and Python lookup tables...')

PLOT_RESULTS = False
ALLOWABLE_NUMERICAL_ERROR = 1e-03

# form an instance of both types of plant
nlplant_c = dynamics.NLPlant(lookup_type='C')
nlplant_py = dynamics.NLPlant(lookup_type='Py')

# retrieve a set of initial conditions and initial inputs from params
x0 = ptu.from_numpy(params.x0)
u0 = ptu.from_numpy(params.u0)

# define the autonomous simulation
def autonomous_sim(nlplant, x0, u, t_end, dt):
    print(f"Using {nlplant.lookup_type} LUTs")
    x_traj = ptu.from_numpy(np.zeros([int(t_end/dt),len(x0)]))
    x = x0
    for idx, t in tqdm(enumerate(np.linspace(start=0,stop=t_end,num=int(t_end/dt)))):
        xdot = nlplant.forward(x0, u)[0]
        x += xdot*dt
        x_traj[idx,:] = x 
    return x_traj

# define number of iterations
t_end = 10
dt = 0.01

# perform both simulations
x_seq_c = autonomous_sim(nlplant_c, torch.clone(x0), torch.clone(u0), t_end, dt)
x_seq_py = autonomous_sim(nlplant_py, torch.clone(x0), torch.clone(u0), t_end, dt)

# create time for plotting
t_seq = np.linspace(start=0, stop=t_end, num=int(t_end/dt))

# push the sequences to numpy for plotting
x_seq_c = ptu.to_numpy(x_seq_c)
x_seq_py = ptu.to_numpy(x_seq_py)

# assert the difference between the simulations is within tolerance
assert np.abs(x_seq_c - x_seq_py).max() < ALLOWABLE_NUMERICAL_ERROR

print('Passed end to end tables test')

if PLOT_RESULTS:
    # plotting functionality
    f0,(f0ax0,f0ax1,f0ax2) = plt.subplots(1, 3)
    f0ax0.plot(t_seq, x_seq_c[:,0], label='npos')
    f0ax0.plot(t_seq, x_seq_py[:,0], label='npos')
    f0ax0.legend(loc="upper right")
    f0ax1.plot(t_seq, x_seq_c[:,1], label='epos')
    f0ax1.plot(t_seq, x_seq_py[:,1], label='epos')
    f0ax1.legend(loc="upper right")
    f0ax2.plot(t_seq, x_seq_c[:,2], label='alt')
    f0ax2.plot(t_seq, x_seq_py[:,2], label='alt')
    f0ax2.legend(loc="upper right")
    f1,(f1ax0,f1ax1,f1ax2) = plt.subplots(1, 3)
    f1ax0.plot(t_seq, x_seq_c[:,3], label='phi')
    f1ax0.plot(t_seq, x_seq_py[:,3], label='phi')
    f1ax0.legend(loc="upper right")
    f1ax1.plot(t_seq, x_seq_c[:,4], label='theta')
    f1ax1.plot(t_seq, x_seq_py[:,4], label='theta')
    f1ax1.legend(loc="upper right")
    f1ax2.plot(t_seq, x_seq_c[:,5], label='psi')
    f1ax2.plot(t_seq, x_seq_py[:,5], label='psi')
    f1ax2.legend(loc="upper right")
    f2,(f2ax0,f2ax1,f2ax2) = plt.subplots(1, 3)
    f2ax0.plot(t_seq, x_seq_c[:,6], label='V')
    f2ax0.plot(t_seq, x_seq_py[:,6], label='V')
    f2ax0.legend(loc="upper right")
    f2ax1.plot(t_seq, x_seq_c[:,7], label='alpha')
    f2ax1.plot(t_seq, x_seq_py[:,7], label='alpha')
    f2ax1.legend(loc="upper right")
    f2ax2.plot(t_seq, x_seq_c[:,8], label='beta')
    f2ax2.plot(t_seq, x_seq_py[:,8], label='beta')
    f2ax2.legend(loc="upper right")
    f3,(f3ax0,f3ax1,f3ax2) = plt.subplots(1, 3)
    f3ax0.plot(t_seq, x_seq_c[:,9], label='p')
    f3ax0.plot(t_seq, x_seq_py[:,9], label='p')
    f3ax0.legend(loc="upper right")
    f3ax1.plot(t_seq, x_seq_c[:,10], label='q')
    f3ax1.plot(t_seq, x_seq_py[:,10], label='q')
    f3ax1.legend(loc="upper right")
    f3ax2.plot(t_seq, x_seq_c[:,11], label='r')
    f3ax2.plot(t_seq, x_seq_py[:,11], label='r')
    f3ax2.legend(loc="upper right")
    plt.show()
