"""
This script validates the trim function found in trim.py by comparing its 
output to that of the MATLAB at the same points.
"""

import dynamics
import params
import trim
import pickle

h_t = 10000 # ft
v_t = 700 # ft/s
x = params.x0
u = params.u0
nlplant = dynamics.NLPlant()

x_trim, opt_info = trim.trim(h_t, v_t, x, u, nlplant)

def load_pkl(dir_path, filename):

    # open the file
    file_to_read = open(dir_path + filename + '.pkl', "rb")
    # load the pickle
    data = pickle.load(file_to_read)
    file_to_read.close()
    return data

dir_path = 'tests/MATLAB_level_flight_trim_reference_states/'
filename = 'alt_10000ft_vt_700fts_xcgr35_xcg30'

ref_data = load_pkl(dir_path, filename)
ref_x_trim = ref_data['trim_state'].squeeze()

trim_state_delta = x_trim - ref_x_trim 

import ipdb
ipdb.set_trace()