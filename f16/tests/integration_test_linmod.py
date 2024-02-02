"""
This script is to compare the ideal linearisations done by the MATLAB and the 
Python.
"""

import os
import numpy as np
import pickle
import linmod
import dynamics
import params
import pytorch_utils as ptu
import torch

print('Performing linearisation integration test...')

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

dir_path = 'tests/MATLAB_level_flight_trim_reference_states/'
filename = 'alt_10000ft_vt_700fts_xcgr35_xcg30'
data = load_pkl(dir_path, filename)
nlplant = dynamics.NLPlant(lookup_type='Py') # lookup_type is Py for torch jacobian
A,B,C,D = linmod.linmod_numerical(nlplant, ptu.from_numpy(params.x0), ptu.from_numpy(params.u0), eps=1e-2)

import ipdb
ipdb.set_trace()
torch.autograd.set_detect_anomaly(True)

x0, u0 = (ptu.from_numpy(params.x0), ptu.from_numpy(params.u0))
At, Bt, Ct, Dt = linmod.linmod_jacobian(nlplant, x0, u0, differentiable=False)

print((A-data['A_hi']).max())
print((At-data['A_hi']).max())


if __name__ == '__main__':
    
    dir_path = 'tests/MATLAB_level_flight_trim_reference_states/'
    convert_matlab = False
    if convert_matlab:
        # do this for every file u want
        for filename in os.listdir(dir_path):
            # check only text files
            if filename.endswith('.mat'):
                filename_no_ext = os.path.splitext(filename)[0]
                mat2pkl(dir_path, filename_no_ext) 

    filename = 'alt_10000ft_vt_700fts_xcgr35_xcg30'
    data = load_pkl(dir_path, filename)

    import ipdb
    ipdb.set_trace()