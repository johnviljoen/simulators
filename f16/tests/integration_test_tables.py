"""
This script iterates through all possible values of the LUTs and checks the difference
between the python and the C implementations. An acceptable discrepancy is at maximum 1e-4
as the tables are only written to 4 significant figures at minimum anyway.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import dynamics
import pytorch_utils as ptu
import params

ALLOWABLE_NUMERICAL_ERROR = 1e-04

print('Performing table integration test...')

nlplant = dynamics.NLPlant() # doesnt matter what lookup_type is

# 20 alpha values in ALPHA1
# 19 beta values in BETA1
# 5 dh values in DH1
# 44 tables

# instantiate an array to track discrepancies
diff = ptu.from_numpy(np.zeros([20,19,5]))

# iterate through all possible lookup table datapoints
for i, alpha in tqdm(enumerate(nlplant.py_lookup.axes['ALPHA1'])):
    for j, beta in enumerate(nlplant.py_lookup.axes['BETA1']):
        for k, dh in enumerate(nlplant.py_lookup.axes['DH1']):

            # pull the alpha beta dh values onto correct device in pytorch
            alpha = ptu.from_numpy(np.array(alpha))
            beta = ptu.from_numpy(np.array(beta))
            dh = ptu.from_numpy(np.array(dh))

            # execute lookup on both C and Python implementations
            c_out = nlplant.c_lut_wrap(alpha,beta,dh)
            py_out = nlplant.py_lut_wrap(alpha,beta,dh)

            # get the difference for each coefficient (they come in
            # sets when using c_lut_wrap and py_lut_wrap - this was 
            # to maintain consistency with the original C)
            coeff_diff = []
            for l, coeff_set in enumerate(c_out):
                for m, coeff in enumerate(coeff_set):
                    coeff_diff.append(c_out[l][m] - py_out[l][m])

            # save the maximum difference to the running difference array
            diff[i,j,k] = max(coeff_diff).squeeze(0).squeeze(0)

# assert 
assert diff.max() < ALLOWABLE_NUMERICAL_ERROR

print('Passed table integration test')