"""
This big scary looking file is entirely for parsing and wrapping the
lookup tables in the aerodata directory. Do not fear, there are only
three classes in this file:

1. CLookup
2. PyLookupNumpy (nonfunctional right now)
3. PyLookupTorch

CLookup uses the original C function of hifi_F16_AeroData.c to parse
and interpolate the lookup tables

PyLookupNumpy and PyLookupTorch use a remade version of this c code
in Python as you might expect from the names. The reason for this is
that PyLookupTorch allows for the lookup tables to be themselves
differentiable, allowing for continuous linearisation using the PyTorch
graph, as well as model-based RL techniques.

PyLookupNumpy was built before PyLookupTorch, and while it is about 5x
faster than PyLookupTorch, it was a stepping stone for me to work out
bugs before building the Torch version.

PyLookupNumpy is currently broken as 'dynamics.py' expects to pass Torch
Tensors to the lookup table parsers, if PyLookupNumpy were modified to
accept and transform tensors to numpy arrays before operating on them it
would work again.

There is a lot of reused code here again, but despite it the classes are 
still very unreadable - I hope to fix this is future with a refactor (said
everyone ever and they never do lmao)

Further the interpolation in the Python lookup tables can be massively
accelerated on GPU if I changed the operation into matrix multiplies,
I just havent gotten round to it yet.
"""

import numpy as np
import ctypes
import os
import common.pytorch_utils as ptu

from f16.parameters import c2f, aerodata_path

tables = ctypes.CDLL(aerodata_path + "/hifi_F16_AeroData.so")

class CLookup():

    def __init__(self):
        pass

    def hifi_C(self, inp):
        
        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0]))
        beta_compat = ctypes.c_double(float(inp[1]))
        el_compat = ctypes.c_double(float(inp[2]))

        tables.hifi_C(alpha_compat, beta_compat, el_compat, retVal_pointer)
        
        return ptu.from_numpy(retVal) # Cx, Cz, Cm, Cy, Cn, Cl

    def hifi_damping(self, inp):
        # this is the one that contains Clr at index 4 
        retVal = np.zeros(9)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0]))
        
        tables.hifi_damping(alpha_compat, retVal_pointer)

        return ptu.from_numpy(retVal) 

    def hifi_C_lef(self, inp):
        
        ''' This table only accepts alpha up to 45 '''
        inp[0] = np.clip(inp[0], a_min=-20., a_max=45.)
        
        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0]))
        beta_compat = ctypes.c_double(float(inp[1]))
        
        tables.hifi_C_lef(alpha_compat, beta_compat, retVal_pointer)
        
        return ptu.from_numpy(retVal)
        
    def hifi_damping_lef(self, inp):
        ''' 
        This table only accepts alpha up to 45
            delta_Cxq_lef
            delta_Cyr_lef
            delta_Cyp_lef
            delta_Czq_lef
            delta_Clr_lef
            delta_Clp_lef
            delta_Cmq_lef
            delta_Cnr_lef
            delta_Cnp_lef
        '''
        inp[0] = np.clip(inp[0], a_min=-20., a_max=45.)
       
        
        retVal = np.zeros(9)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0]))
        
        tables.hifi_damping_lef(alpha_compat, retVal_pointer)
        
        return ptu.from_numpy(retVal)

    def hifi_rudder(self, inp):
        
        retVal = np.zeros(3)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0]))
        beta_compat = ctypes.c_double(float(inp[1]))
        
        tables.hifi_rudder(alpha_compat, beta_compat, retVal_pointer)
        
        return ptu.from_numpy(retVal)

    def hifi_ailerons(self, inp):
        
        ''' This table only accepts alpha up to 45 '''
        inp[0] = np.clip(inp[0], a_min=-20., a_max=45.)
        
        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0]))
        beta_compat = ctypes.c_double(float(inp[1]))
        
        tables.hifi_ailerons(alpha_compat, beta_compat, retVal_pointer)
        
        return ptu.from_numpy(retVal)

    def hifi_other_coeffs(self, inp):
        
        '''expects an input of alpha, el'''
        
        retVal = np.zeros(5)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0]))
        el_compat = ctypes.c_double(float(inp[1]))
        
        tables.hifi_other_coeffs(alpha_compat, el_compat, retVal_pointer)
        
        retVal[4] = 0 # ignore deep-stall regime, delta_Cm_ds = 0
        
        return ptu.from_numpy(retVal)

class PyLookupNumpy():
    """
    This class parses the .dat files into a dictionary of tensors of the correct dimensions,
    which can be accessed by the key of the filename from which the lookup values were read.
    
    This class then takes the parsed data and does some calculations on it to form the final
    LUT values which are also then interpolated.
    """

    def __init__(self):
        # indices lookup
        self.axes = {}
        self.axes['ALPHA1'] = np.array(self.read_file("aerodata/ALPHA1.dat"))
        self.axes['ALPHA2'] = np.array(self.read_file("aerodata/ALPHA2.dat"))
        self.axes['BETA1'] = np.array(self.read_file("aerodata/BETA1.dat"))
        self.axes['DH1'] = np.array(self.read_file("aerodata/DH1.dat"))
        self.axes['DH2'] = np.array(self.read_file("aerodata/DH2.dat"))
        
        # tables store the actual data, points are the alpha, beta, dh axes 
        self.tables = {}
        self.points = {}
        self.ndinfo = {}
        for file in os.listdir("aerodata"):
            if file == 'CM1020_ALPHA1_103.dat':
                continue
            alpha_len = None
            beta_len = None
            dh_len = None
            alpha_fi = None
            beta_fi = None
            dh_fi = None
            if "_ALPHA1" in file:
                alpha_len = len(self.axes['ALPHA1'])
                alpha_fi = 'ALPHA1'
            if "_ALPHA2" in file:
                alpha_len = len(self.axes['ALPHA2'])
                alpha_fi = 'ALPHA2'
            if "_BETA1" in file:
                beta_len = len(self.axes['BETA1'])
                beta_fi = 'BETA1'
            if "_DH1" in file:
                dh_len = len(self.axes['DH1'])
                dh_fi = 'DH1'
            if "_DH2" in file:
                dh_len = len(self.axes['DH2'])
                dh_fi = 'DH2'

            temp = [alpha_len, beta_len, dh_len]
            dims = [i for i in temp if i is not None]
            
            self.ndinfo[file] = {
                'alpha_fi': alpha_fi,
                'beta_fi': beta_fi,
                'dh_fi': dh_fi
            }

            # 1D tables
            if len(dims) == 1:
                self.tables[file] = np.array(self.read_file(f"aerodata/{file}"))
                if file == "ETA_DH1_brett.dat":
                    self.points[file] = (self.axes[self.ndinfo[file]['dh_fi']]) 
                else:
                    self.points[file] = (self.axes[self.ndinfo[file]['alpha_fi']]) 
                    
         
            # 2D tables
            elif len(dims) == 2:
                self.tables[file] = np.array(self.read_file(f"aerodata/{file}")).reshape([dims[0],dims[1]],order='F')
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']])

            # 3D tables
            elif len(dims) == 3:
                self.tables[file] = np.array(self.read_file(f"aerodata/{file}")).reshape([dims[0],dims[1],dims[2]],order='F')
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']],
                    self.axes[self.ndinfo[file]['dh_fi']]) 

    def read_file(self, path):
        """
        Utility for reading in the .dat files that comprise all of the aerodata
        """
        
        # get the indices of the various tables first
        with open(path) as f:
            lines = f.readlines()
        temp = lines[0][:-1].split()
        line = [float(i) for i in temp]
        return line

    def get_bounds_1d(self, inp, coeff):
        # can be either alpha or el

        ndinfo = self.ndinfo[c2f[coeff]]
        try:
            assert ndinfo['alpha_fi'] is not None
            alpha_ax = self.axes[ndinfo['alpha_fi']]
            out0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
            out1 = out0 + 1

        except:
            assert ndinfo['dh_fi'] is not None
            dh_ax = self.axes[ndinfo['dh_fi']]
            out0 = len([i for i in dh_ax-inp[-1] if i < 0]) - 1
            out1 = out0 + 1
        return out0, out1



    def get_bounds_2d(self, inp, coeff):
        # inp expected in form [alpha, beta]

        ndinfo = self.ndinfo[c2f[coeff]]

        alpha_ax = self.axes[ndinfo['alpha_fi']]
        beta_ax = self.axes[ndinfo['beta_fi']]

        alpha0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
        alpha1 = alpha0 + 1

        beta0 = len([i for i in beta_ax-inp[1] if i < 0]) - 1
        beta1 = beta0 + 1

        return alpha0, alpha1, beta0, beta1


    def get_bounds_3d(self, inp, coeff):
        
        # this is for getting the indices of the alpha, beta, el values around the inp
        # table = py_parse.tables[c2f[table_name]]

        # inp expected in form [alpha, beta, el]

        ndinfo = self.ndinfo[c2f[coeff]]

        alpha_ax = self.axes[ndinfo['alpha_fi']]
        beta_ax = self.axes[ndinfo['beta_fi']]
        dh_ax = self.axes[ndinfo['dh_fi']]

        alpha0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
        alpha1 = alpha0 + 1

        beta0 = len([i for i in beta_ax-inp[1] if i < 0]) - 1
        beta1 = beta0 + 1
        
        dh0 = len([i for i in dh_ax-inp[2] if i < 0]) - 1
        dh1 = dh0 + 1

        return alpha0, alpha1, beta0, beta1, dh0, dh1
        
    
    def interp_1d(self, inp, coeff):
        # here x0, x1 refer to alpha0, alpha1 usually (one exception where it is dh)
        # and y0, y1 refer to the coefficent values at these values of alpha
        if self.ndinfo[c2f[coeff]]['dh_fi'] is None:
            table = self.axes[self.ndinfo[c2f[coeff]]['alpha_fi']]
        else:
            table = self.axes[self.ndinfo[c2f[coeff]]['dh_fi']]
        
        # inp_true = inp
        # inp_clip = np.clip(inp, a_min=table.min(), a_max=table.max())
        # choice of using clip or true here dictate 1st order extrapolation or 0th order extrapolation
        x = inp[0]

        x0_idx, x1_idx = self.get_bounds_1d(inp, coeff)
        
        try:
            x0 = table[x0_idx]
            x1 = table[x1_idx]
        except:
            raise Exception(f'input to {coeff} table out of bounds, check alpha and dh')

        #x0 = self.axes[c2f[coeff]]
        y0 = self.tables[c2f[coeff]][x0_idx]
        y1 = self.tables[c2f[coeff]][x1_idx]

        C = np.array(y0*(x1 - x)/(x1 - x0) + y1*(x - x0)/(x1 - x0))
        return ptu.from_numpy(C)

    def interp_2d(self, inp, coeff):
        
        alpha_table = self.axes[self.ndinfo[c2f[coeff]]['alpha_fi']]
        beta_table = self.axes[self.ndinfo[c2f[coeff]]['beta_fi']]

        # inp_true = inp 
        # inp_clip = np.clip(inp, a_min=[alpha_table.min(),beta_table.min()], a_max=[alpha_table.max(),beta_table.max()])
        # choice of using clip or true here dictate 1st order extrapolation or 0th order extrapolation
        x = inp[0]
        y = inp[1]
        
        x0_idx, x1_idx, y0_idx, y1_idx = self.get_bounds_2d(inp, coeff)

        try:
            x0 = alpha_table[x0_idx]
            x1 = alpha_table[x1_idx]
            y0 = beta_table[y0_idx]
            y1 = beta_table[y1_idx]
        except:
            raise Exception(f'input to {coeff} out of bounds, check alpha and/or beta')

        C00 = self.tables[c2f[coeff]][x0_idx, y0_idx]
        C01 = self.tables[c2f[coeff]][x0_idx, y1_idx]
        C10 = self.tables[c2f[coeff]][x1_idx, y0_idx]
        C11 = self.tables[c2f[coeff]][x1_idx, y1_idx]

        # chose not to use the linear algebra route, but instead the line by line route
        # as it allows for more clarity I think, as well as leaving the final result in
        # a nice, easy to use 0D array for ptu.from_numpy to operate on
        # C = 1/((x1 - x0)*(y1 - y0)) * np.array([[x1 - x, x - x0]]) @ np.array([[Q00, Q01],[Q10, Q11]]) @ np.array([[y1 - y],[y - y0]])

        xd = (x - x0)/(x1 - x0)
        yd = (y - y0)/(y1 - y0)

        C0 = C00 * (1 - xd) + C10 * xd
        C1 = C01 * (1 - xd) + C11 * xd

        C = C0 * (1 - yd) + C1 * yd

        C = np.array(C)
        
        return ptu.from_numpy(C)

    def interp_3d(self, inp, coeff):
        x0_idx, x1_idx, y0_idx, y1_idx, z0_idx, z1_idx = self.get_bounds_3d(inp, coeff)

        x = inp[0]
        y = inp[1]
        z = inp[2]

        alpha_table = self.axes[self.ndinfo[c2f[coeff]]['alpha_fi']]
        beta_table = self.axes[self.ndinfo[c2f[coeff]]['beta_fi']]
        dh_table = self.axes[self.ndinfo[c2f[coeff]]['dh_fi']]
        
        try:
            x0 = alpha_table[x0_idx]
            x1 = alpha_table[x1_idx]
            y0 = beta_table[y0_idx]
            y1 = beta_table[y1_idx]
            z0 = dh_table[z0_idx]
            z1 = dh_table[z1_idx]
        except:
            raise Exception(f'input to {coeff} out of bounds, check alpha, beta and dh')

        # differences in x y z from desired positions 
        xd = (x - x0)/(x1 - x0)
        yd = (y - y0)/(y1 - y0)
        zd = (z - z0)/(z1 - z0)

        # at the base dh level
        C000 = self.tables[c2f[coeff]][x0_idx, y0_idx, z0_idx]
        C010 = self.tables[c2f[coeff]][x0_idx, y1_idx, z0_idx]
        C100 = self.tables[c2f[coeff]][x1_idx, y0_idx, z0_idx]
        C110 = self.tables[c2f[coeff]][x1_idx, y1_idx, z0_idx]
       
        # at the top dh level
        C001 = self.tables[c2f[coeff]][x0_idx, y0_idx, z1_idx]
        C011 = self.tables[c2f[coeff]][x0_idx, y1_idx, z1_idx]
        C101 = self.tables[c2f[coeff]][x1_idx, y0_idx, z1_idx]
        C111 = self.tables[c2f[coeff]][x1_idx, y1_idx, z1_idx]

        # 
        C00 = C000 * (1 - xd) + C100 * xd
        C01 = C001 * (1 - xd) + C101 * xd
        C10 = C010 * (1 - xd) + C110 * xd
        C11 = C011 * (1 - xd) + C111 * xd

        #
        C0 = C00 * (1 - yd) + C10 * yd
        C1 = C01 * (1 - yd) + C11 * yd

        # 
        C = C0 * (1 - zd) + C1 * zd

        C = np.array(C)


        return ptu.from_numpy(C)

class PyLookupTorch():
    """
    This class parses the .dat files into a dictionary of tensors of the correct dimensions,
    which can be accessed by the key of the filename from which the lookup values were read.
    
    This class then takes the parsed data and does some calculations on it to form the final
    LUT values which are also then interpolated.
    """

    def __init__(self):
        # indices lookup
        self.axes = {}
        self.axes['ALPHA1'] = ptu.from_numpy(np.array(self.read_file("aerodata/ALPHA1.dat")))
        self.axes['ALPHA2'] = ptu.from_numpy(np.array(self.read_file("aerodata/ALPHA2.dat")))
        self.axes['BETA1'] = ptu.from_numpy(np.array(self.read_file("aerodata/BETA1.dat")))
        self.axes['DH1'] = ptu.from_numpy(np.array(self.read_file("aerodata/DH1.dat")))
        self.axes['DH2'] = ptu.from_numpy(np.array(self.read_file("aerodata/DH2.dat")))
        
        # tables store the actual data, points are the alpha, beta, dh axes 
        self.tables = {}
        self.points = {}
        self.ndinfo = {}
        for file in os.listdir("aerodata"):
            if file == 'CM1020_ALPHA1_103.dat':
                continue
            alpha_len = None
            beta_len = None
            dh_len = None
            alpha_fi = None
            beta_fi = None
            dh_fi = None
            if "_ALPHA1" in file:
                alpha_len = len(self.axes['ALPHA1'])
                alpha_fi = 'ALPHA1'
            if "_ALPHA2" in file:
                alpha_len = len(self.axes['ALPHA2'])
                alpha_fi = 'ALPHA2'
            if "_BETA1" in file:
                beta_len = len(self.axes['BETA1'])
                beta_fi = 'BETA1'
            if "_DH1" in file:
                dh_len = len(self.axes['DH1'])
                dh_fi = 'DH1'
            if "_DH2" in file:
                dh_len = len(self.axes['DH2'])
                dh_fi = 'DH2'

            temp = [alpha_len, beta_len, dh_len]
            dims = [i for i in temp if i is not None]
            
            self.ndinfo[file] = {
                'alpha_fi': alpha_fi,
                'beta_fi': beta_fi,
                'dh_fi': dh_fi
            }

            # 1D tables
            if len(dims) == 1:
                self.tables[file] = ptu.from_numpy(np.array(self.read_file(f"aerodata/{file}")))
                if file == "ETA_DH1_brett.dat":
                    self.points[file] = (self.axes[self.ndinfo[file]['dh_fi']]) 
                else:
                    self.points[file] = (self.axes[self.ndinfo[file]['alpha_fi']]) 
                    
         
            # 2D tables
            elif len(dims) == 2:
                self.tables[file] = ptu.from_numpy(np.array(self.read_file(f"aerodata/{file}")).reshape([dims[0],dims[1]],order='F'))
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']])

            # 3D tables
            elif len(dims) == 3:
                self.tables[file] = ptu.from_numpy(np.array(self.read_file(f"aerodata/{file}")).reshape([dims[0],dims[1],dims[2]],order='F'))
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']],
                    self.axes[self.ndinfo[file]['dh_fi']]) 

    def read_file(self, path):
        """
        Utility for reading in the .dat files that comprise all of the aerodata
        """
        
        # get the indices of the various tables first
        with open(path) as f:
            lines = f.readlines()
        temp = lines[0][:-1].split()
        line = [float(i) for i in temp]
        return line

    def get_bounds_1d(self, inp, coeff):
        # can be either alpha or el

        ndinfo = self.ndinfo[c2f[coeff]]
        try:
            assert ndinfo['alpha_fi'] is not None
            alpha_ax = self.axes[ndinfo['alpha_fi']]
            out0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
            out1 = out0 + 1

        except:
            assert ndinfo['dh_fi'] is not None
            dh_ax = self.axes[ndinfo['dh_fi']]
            out0 = len([i for i in dh_ax-inp[-1] if i < 0]) - 1
            out1 = out0 + 1
        return out0, out1



    def get_bounds_2d(self, inp, coeff):
        # inp expected in form [alpha, beta]

        ndinfo = self.ndinfo[c2f[coeff]]

        alpha_ax = self.axes[ndinfo['alpha_fi']]
        beta_ax = self.axes[ndinfo['beta_fi']]

        alpha0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
        alpha1 = alpha0 + 1

        beta0 = len([i for i in beta_ax-inp[1] if i < 0]) - 1
        beta1 = beta0 + 1

        return alpha0, alpha1, beta0, beta1


    def get_bounds_3d(self, inp, coeff):
        
        # this is for getting the indices of the alpha, beta, el values around the inp
        # table = py_parse.tables[c2f[table_name]]

        # inp expected in form [alpha, beta, el]

        ndinfo = self.ndinfo[c2f[coeff]]

        alpha_ax = self.axes[ndinfo['alpha_fi']]
        beta_ax = self.axes[ndinfo['beta_fi']]
        dh_ax = self.axes[ndinfo['dh_fi']]

        alpha0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
        alpha1 = alpha0 + 1

        beta0 = len([i for i in beta_ax-inp[1] if i < 0]) - 1
        beta1 = beta0 + 1
        
        dh0 = len([i for i in dh_ax-inp[2] if i < 0]) - 1
        dh1 = dh0 + 1

        return alpha0, alpha1, beta0, beta1, dh0, dh1
        
    
    def interp_1d(self, inp, coeff):
        # here x0, x1 refer to alpha0, alpha1 usually (one exception where it is dh)
        # and y0, y1 refer to the coefficent values at these values of alpha
        if self.ndinfo[c2f[coeff]]['dh_fi'] is None:
            table = self.axes[self.ndinfo[c2f[coeff]]['alpha_fi']]
        else:
            table = self.axes[self.ndinfo[c2f[coeff]]['dh_fi']]
        
        # inp_true = inp
        # inp_clip = np.clip(inp, a_min=table.min(), a_max=table.max())
        # choice of using clip or true here dictate 1st order extrapolation or 0th order extrapolation
        x = inp[0]

        x0_idx, x1_idx = self.get_bounds_1d(inp, coeff)
        
        try:
            x0 = table[x0_idx]
            x1 = table[x1_idx]
        except:
            raise Exception(f'input to {coeff} table out of bounds, check alpha and dh')

        #x0 = self.axes[c2f[coeff]]
        y0 = self.tables[c2f[coeff]][x0_idx]
        y1 = self.tables[c2f[coeff]][x1_idx]

        C = y0*(x1 - x)/(x1 - x0) + y1*(x - x0)/(x1 - x0)
        return C

    def interp_2d(self, inp, coeff):
        
        alpha_table = self.axes[self.ndinfo[c2f[coeff]]['alpha_fi']]
        beta_table = self.axes[self.ndinfo[c2f[coeff]]['beta_fi']]

        # inp_true = inp 
        # inp_clip = np.clip(inp, a_min=[alpha_table.min(),beta_table.min()], a_max=[alpha_table.max(),beta_table.max()])
        # choice of using clip or true here dictate 1st order extrapolation or 0th order extrapolation
        x = inp[0]
        y = inp[1]
        
        x0_idx, x1_idx, y0_idx, y1_idx = self.get_bounds_2d(inp, coeff)

        try:
            x0 = alpha_table[x0_idx]
            x1 = alpha_table[x1_idx]
            y0 = beta_table[y0_idx]
            y1 = beta_table[y1_idx]
        except:
            raise Exception(f'input to {coeff} out of bounds, check alpha and/or beta')

        C00 = self.tables[c2f[coeff]][x0_idx, y0_idx]
        C01 = self.tables[c2f[coeff]][x0_idx, y1_idx]
        C10 = self.tables[c2f[coeff]][x1_idx, y0_idx]
        C11 = self.tables[c2f[coeff]][x1_idx, y1_idx]

        # chose not to use the linear algebra route, but instead the line by line route
        # as it allows for more clarity I think, as well as leaving the final result in
        # a nice, easy to use 0D array for ptu.from_numpy to operate on
        # C = 1/((x1 - x0)*(y1 - y0)) * np.array([[x1 - x, x - x0]]) @ np.array([[Q00, Q01],[Q10, Q11]]) @ np.array([[y1 - y],[y - y0]])

        xd = (x - x0)/(x1 - x0)
        yd = (y - y0)/(y1 - y0)

        C0 = C00 * (1 - xd) + C10 * xd
        C1 = C01 * (1 - xd) + C11 * xd

        C = C0 * (1 - yd) + C1 * yd
        
        return C

    def interp_3d(self, inp, coeff):
        x0_idx, x1_idx, y0_idx, y1_idx, z0_idx, z1_idx = self.get_bounds_3d(inp, coeff)

        x = inp[0]
        y = inp[1]
        z = inp[2]

        alpha_table = self.axes[self.ndinfo[c2f[coeff]]['alpha_fi']]
        beta_table = self.axes[self.ndinfo[c2f[coeff]]['beta_fi']]
        dh_table = self.axes[self.ndinfo[c2f[coeff]]['dh_fi']]
        
        try:
            x0 = alpha_table[x0_idx]
            x1 = alpha_table[x1_idx]
            y0 = beta_table[y0_idx]
            y1 = beta_table[y1_idx]
            z0 = dh_table[z0_idx]
            z1 = dh_table[z1_idx]
        except:
            raise Exception(f'input to {coeff} out of bounds, check alpha, beta and dh')

        # differences in x y z from desired positions 
        xd = (x - x0)/(x1 - x0)
        yd = (y - y0)/(y1 - y0)
        zd = (z - z0)/(z1 - z0)

        # at the base dh level
        C000 = self.tables[c2f[coeff]][x0_idx, y0_idx, z0_idx]
        C010 = self.tables[c2f[coeff]][x0_idx, y1_idx, z0_idx]
        C100 = self.tables[c2f[coeff]][x1_idx, y0_idx, z0_idx]
        C110 = self.tables[c2f[coeff]][x1_idx, y1_idx, z0_idx]
       
        # at the top dh level
        C001 = self.tables[c2f[coeff]][x0_idx, y0_idx, z1_idx]
        C011 = self.tables[c2f[coeff]][x0_idx, y1_idx, z1_idx]
        C101 = self.tables[c2f[coeff]][x1_idx, y0_idx, z1_idx]
        C111 = self.tables[c2f[coeff]][x1_idx, y1_idx, z1_idx]

        # 
        C00 = C000 * (1 - xd) + C100 * xd
        C01 = C001 * (1 - xd) + C101 * xd
        C10 = C010 * (1 - xd) + C110 * xd
        C11 = C011 * (1 - xd) + C111 * xd

        #
        C0 = C00 * (1 - yd) + C10 * yd
        C1 = C01 * (1 - yd) + C11 * yd

        # 
        C = C0 * (1 - zd) + C1 * zd

        return C