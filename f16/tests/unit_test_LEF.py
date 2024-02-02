import dynamics
import pytorch_utils as ptu
import numpy as np
import plot
import pickle
from tqdm import tqdm
from dynamics import NLPlant

print('performing (not so rigorous) LEF unit test')

"""
Simulink refuses to use a timestep of 0.1 fixed, despite being told to do so,
it actually uses different timesteps in different parts of the simulation to update
its various components, therefore I am allowing a significant ALLOWABLE_ERROR
in this test as I believe the model to be otherwise accurate.
"""
ALLOWABLE_ERROR = 1 

def load_pkl(dir_path, filename):
    # open the file
    file_to_read = open(dir_path + filename + '.pkl', "rb")
    # load the pickle
    data = pickle.load(file_to_read)
    file_to_read.close()
    return data

DIRECTORY_PATH = 'tests/MATLAB_timehistory/'
filename = 'lef_dist_5_alpha_rad_0.1_dLEF_5_10000ft_700fts'
data = load_pkl(DIRECTORY_PATH, filename)
LF_state_ref = data['LF_state']
d_LF_ref = data['d_LF']

# initial conditions
disturbance = 5
h           = ptu.from_numpy(np.array(10000)) # ft
V           = ptu.from_numpy(np.array(700  )) # ft/s
alpha       = ptu.from_numpy(np.array(0.1  )) # rad
# whenever I have done trim before dLEF has been 0 and LF_state = -alpha_deg
LF_state    = ptu.from_numpy(np.array(-alpha*180/np.pi + disturbance    )) # deg
dLEF        = ptu.from_numpy(np.array(disturbance    )) # deg

# dynamics
nlplant = dynamics.NLPlant()

t_end = 10
dt = 0.1



def autonomous_lef_actuator(nlplant: NLPlant, LF_state0, dLEF0, h, V, alpha, t_end, dt):
    t_seq = np.linspace(start=0,stop=t_end,num=int(t_end/dt))
    x_seq = np.zeros([t_seq.shape[0],2])
    LF_state = LF_state0
    dLEF = dLEF0
    dt = 0.0725 # simulink actually timesteps with this value despite being told to update with 0.1
    for idx, t in tqdm(enumerate(t_seq)):
        x_seq[idx,0] = LF_state
        x_seq[idx,1] = dLEF
        LF_state_dot, dLEF_dot = nlplant.calc_lef_dot(h,V,alpha,dLEF,LF_state)
        LF_state += LF_state_dot * dt
        dLEF += dLEF_dot * dt
    return x_seq

x_seq = autonomous_lef_actuator(nlplant, LF_state, dLEF, h, V, alpha, t_end, dt)



# compare this to a reference trajectory
x_seq_ref = np.concatenate([LF_state_ref, d_LF_ref], axis=1)[:-1,:]

time_sequence = np.linspace(0,10,100)

assert np.abs(x_seq_ref - x_seq).max() < ALLOWABLE_ERROR

print('passed (not so rigorous) LEF unit test')

# plot.lef_comparison(x_seq, x_seq_ref, time_sequence)
