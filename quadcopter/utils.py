import numpy as np
import torch
import casadi as ca

import common.pytorch_utils as ptu

class makeMixerFM:

    # Motor 1 is front left, then clockwise numbering.
    # A mixer like this one allows to find the exact RPM of each motor 
    # given a desired thrust and desired moments.
    # Inspiration for this mixer (or coefficient matrix) and how it is used : 
    # https://link.springer.com/article/10.1007/s13369-017-2433-2 (https://sci-hub.tw/10.1007/s13369-017-2433-2)
    
    @staticmethod
    def pytorch(params):
        dxm = params["dxm"]
        dym = params["dym"]
        kTh = params["kTh"]
        kTo = params["kTo"] 

        return ptu.tensor([[    kTh,      kTh,      kTh,      kTh],
                            [dym*kTh, -dym*kTh,  -dym*kTh, dym*kTh],
                            [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                            [   -kTo,      kTo,     -kTo,      kTo]])

    @staticmethod
    def numpy(params):
        dxm = params["dxm"]
        dym = params["dym"]
        kTh = params["kTh"]
        kTo = params["kTo"] 

        return np.array([[    kTh,      kTh,      kTh,      kTh],
                        [dym*kTh, -dym*kTh,  -dym*kTh, dym*kTh],
                        [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                        [   -kTo,      kTo,     -kTo,      kTo]])

class applyMixerFM:

    @staticmethod
    def numpy(params, thr, moment):   
        t = np.array([thr, moment[0], moment[1], moment[2]])
        w_cmd = np.sqrt(np.clip(np.dot(params["mixerFMinv"], t), params["minWmotor"]**2, params["maxWmotor"]**2))
        return w_cmd

    @staticmethod
    def pytorch(params, thr, moment):
        t = torch.stack([thr, moment[0], moment[1], moment[2]])
        w_cmd = torch.sqrt(torch.clip(params["mixerFMinv"] @ t, params["minWmotor"]**2, params["maxWmotor"]**2))
        return w_cmd
    
    @staticmethod
    def pytorch_vectorized(params, thr, moment):
        t = torch.stack([thr, moment[:,0], moment[:,1], moment[:,2]], dim=1)

        # Perform batched matrix multiplication
        w_cmd_sq = torch.bmm(params["mixerFMinv"].unsqueeze(0).expand(t.size(0), -1, -1), t.unsqueeze(-1)).squeeze(-1)

        # Clip and take square root
        w_cmd = torch.sqrt(torch.clamp(w_cmd_sq, min=params["minWmotor"]**2, max=params["maxWmotor"]**2))
        
        # w_cmd = torch.sqrt(torch.clip(params["mixerFMinv"] @ t, params["minWmotor"]**2, params["maxWmotor"]**2))
        return w_cmd
    
    @staticmethod
    def casadi(params, thr, moment):
        t = ca.vertcat(thr, moment[:,0], moment[:,1], moment[:,2])

        mixer = ca.DM(ptu.to_numpy(params["mixerFMinv"]))
        # Perform matrix multiplication
        w_cmd_sq = ca.mtimes(mixer, t)

        # Clamp and take square root
        w_cmd_sq_clamped = ca.fmax(ca.fmin(w_cmd_sq, params["maxWmotor"]**2), params["minWmotor"]**2)
        w_cmd = ca.sqrt(w_cmd_sq_clamped)
        
        return w_cmd