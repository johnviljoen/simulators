import pickle
import os
import numpy as np
import dynamics
import common.pytorch_utils as ptu
import copy
from tqdm import tqdm

CONVERT_MATLAB = True
DIRECTORY_PATH = 'tests/MATLAB_timehistory/'
ALLOWABLE_ERROR = 5 # largest delta for altitude always, 5ft is nothing RIGHT

"""This section is purely for conversion into MATLAB, which we only
do once, and then save into the directory the pickle so we
never have to interact with .mat files ever again"""



def mat2pkl(dir_path, filename):
    import scipy.io
    # import matlab data
    data = scipy.io.loadmat(dir_path + filename + '.mat')
    # create a binary pickle file 
    f = open(dir_path + filename + '.pkl',"wb")
    # write the python object (dict) to pickle file
    pickle.dump(data,f)
    # close file
    f.close()

def load_pkl(dir_path, filename):
    # open the file
    file_to_read = open(dir_path + filename + '.pkl', "rb")
    # load the pickle
    data = pickle.load(file_to_read)
    file_to_read.close()
    return data

if CONVERT_MATLAB:
    # do this for every file u want
    for filename in os.listdir(DIRECTORY_PATH):
        # check only text files
        if filename.endswith('.mat'):
            filename_no_ext = os.path.splitext(filename)[0]
            mat2pkl(DIRECTORY_PATH, filename_no_ext)

"""The next section is for the actual testing"""

filename = '10000ft_700fts_x_sim'
data = load_pkl(DIRECTORY_PATH, filename)['x_sim']

alpha0 = data[0,7]

# got the data LF_state and dLEF the wrong way around
LF_state0 = copy.deepcopy(data[0,16])
dLEF0 = copy.deepcopy(data[0,17])
data[0,16] = dLEF0
data[0,17] = LF_state0

x0 = ptu.from_numpy(data[0,:])
u0 = ptu.from_numpy(data[0,12:16])

x_seq_ref = data[0:-1,:]
# set initial alpha, theta to be in radians
x0[7] *= np.pi/180
x0[4] *= np.pi/180

t_end = 30
dt = 0.001

nlplant = dynamics.NLPlant()
raw_x_seq_sim = ptu.to_numpy(autonomous(nlplant, x0, u0, t_end, dt))

# create time for plotting
t_seq = np.linspace(start=0, stop=t_end, num=int(t_end/dt))

x_seq_sim = raw_x_seq_sim[::100,:]

# checking input discrepancies between the two systems -> there is none
print(f'maximum input discrepancy: {np.abs(x_seq_sim[:,12:16]-x_seq_ref[:,12:16]).max()}')

empirical_xdot_sim = x_seq_sim[1,:] - x_seq_sim[0,:]
empirical_xdot_ref = x_seq_ref[1,:] - x_seq_ref[0,:]

x_seq_ref[:,3] *= np.pi/180
x_seq_ref[:,4] *= np.pi/180
x_seq_ref[:,5] *= np.pi/180

x_seq_ref[:,7] *= np.pi/180
x_seq_ref[:,8] *= np.pi/180

x_seq_ref[:,9] *= np.pi/180
x_seq_ref[:,10] *= np.pi/180
x_seq_ref[:,11] *= np.pi/180

# and these surfaces go directly into nlplant in the MATLAB
assert np.abs(x_seq_sim - x_seq_ref).max() < ALLOWABLE_ERROR

print('passed end to end autonomous simulation test')
# plot.state_comparison(x_seq_sim, x_seq_ref, t_seq[::100])
