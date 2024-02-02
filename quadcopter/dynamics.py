import copy, os

import numpy as np
import torch
import casadi as ca

import common.pytorch_utils as ptu

from typing import Dict

class state_dot:

    @staticmethod
    def numpy(state: np.ndarray, cmd: np.ndarray, params: Dict):

        q0 =    state[3]
        q1 =    state[4]
        q2 =    state[5]
        q3 =    state[6]
        xdot =  state[7]
        ydot =  state[8]
        zdot =  state[9]
        p =     state[10]
        q =     state[11]
        r =     state[12]
        wM1 =   state[13]
        wM2 =   state[14]
        wM3 =   state[15]
        wM4 =   state[16]

        wMotor = np.stack([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, params["minWmotor"], params["maxWmotor"])
        thrust = params["kTh"] * wMotor ** 2
        torque = params["kTo"] * wMotor ** 2

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = [0]*3

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = np.stack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * np.sign(velW * np.cos(qW1) * np.cos(qW2) - xdot)
                    * (velW * np.cos(qW1) * np.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    params["Cd"]
                    * np.sign(velW * np.sin(qW1) * np.cos(qW2) - ydot)
                    * (velW * np.sin(qW1) * np.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    -params["Cd"] * np.sign(velW * np.sin(qW2) + zdot) * (velW * np.sin(qW2) + zdot) ** 2
                    - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (params["IB"][1,1] - params["IB"][2,2]) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * params["dym"]
                )
                / params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (params["IB"][2,2] - params["IB"][0,0]) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * params["dxm"]
                )
                / params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((params["IB"][0,0] - params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / params["IB"][2,2],
            ]
        )

        # we must limit the actuator rotational rate
        omega_check_upper = state[13:] > params["maxWmotor"]
        omega_check_lower = state[13:] < params["minWmotor"]
        ActuatorsDot = cmd/params["IRzz"]
        ActuatorsDot[(omega_check_upper) | (omega_check_lower)] = 0

        # State Derivative Vector
        # ---------------------------
        if state.shape[0] == 1:
            return np.hstack([DynamicsDot.squeeze(), ActuatorsDot.squeeze()])
        else:
            return np.hstack([DynamicsDot.T, ActuatorsDot])

    @staticmethod
    def casadi(state: ca.MX, cmd: ca.MX, params: Dict):
        # formerly known as QuadcopterCA

        # Import params to numpy for CasADI
        # ---------------------------
        IB = params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]

        # Unpack state tensor for readability
        # ---------------------------
        q0 =    state[3]
        q1 =    state[4]
        q2 =    state[5]
        q3 =    state[6]
        xdot =  state[7]
        ydot =  state[8]
        zdot =  state[9]
        p =     state[10]
        q =     state[11]
        r =     state[12]
        wM1 =   state[13]
        wM2 =   state[14]
        wM3 =   state[15]
        wM4 =   state[16]

        # a tiny bit more readable
        ThrM1 = params["kTh"] * wM1 ** 2
        ThrM2 = params["kTh"] * wM2 ** 2
        ThrM3 = params["kTh"] * wM3 ** 2
        ThrM4 = params["kTh"] * wM4 ** 2
        TorM1 = params["kTo"] * wM1 ** 2
        TorM2 = params["kTo"] * wM2 ** 2
        TorM3 = params["kTo"] * wM3 ** 2
        TorM4 = params["kTo"] * wM4 ** 2

        # Wind Model (zero in expectation)
        # ---------------------------
        velW, qW1, qW2 = 0, 0, 0

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = ca.vertcat(
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    params["Cd"]
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    -params["Cd"] * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (IByy - IBzz) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * params["dym"]
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * params["dxm"]
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0]/params["IRzz"], cmd[1]/params["IRzz"], cmd[2]/params["IRzz"], cmd[3]/params["IRzz"]
        )

        if DynamicsDot.shape[1] == 17:
            print('fin')

        # State Derivative Vector
        # ---------------------------
        return DynamicsDot

    @staticmethod
    def casadi_vectorized(state: ca.MX, cmd: ca.MX, params: Dict):

        # Import params to numpy for CasADI
        # ---------------------------
        IB = params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]

        # Unpack state tensor for readability
        # ---------------------------
        q0 =    state[3,:]
        q1 =    state[4,:]
        q2 =    state[5,:]
        q3 =    state[6,:]
        xdot =  state[7,:]
        ydot =  state[8,:]
        zdot =  state[9,:]
        p =     state[10,:]
        q =     state[11,:]
        r =     state[12,:]
        wM1 =   state[13,:]
        wM2 =   state[14,:]
        wM3 =   state[15,:]
        wM4 =   state[16,:]

        # a tiny bit more readable
        ThrM1 = params["kTh"] * wM1 ** 2
        ThrM2 = params["kTh"] * wM2 ** 2
        ThrM3 = params["kTh"] * wM3 ** 2
        ThrM4 = params["kTh"] * wM4 ** 2
        TorM1 = params["kTo"] * wM1 ** 2
        TorM2 = params["kTo"] * wM2 ** 2
        TorM3 = params["kTo"] * wM3 ** 2
        TorM4 = params["kTo"] * wM4 ** 2

        # Wind Model (zero in expectation)
        # ---------------------------
        velW, qW1, qW2 = 0, 0, 0

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = ca.vertcat(
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    params["Cd"]
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    -params["Cd"] * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (IByy - IBzz) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * params["dym"]
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * params["dxm"]
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0,:]/params["IRzz"], cmd[1,:]/params["IRzz"], cmd[2,:]/params["IRzz"], cmd[3,:]/params["IRzz"]
        )

        # State Derivative Vector
        # ---------------------------
        return DynamicsDot
    
    @staticmethod
    def pytorch_vectorized(state: torch.Tensor, cmd: torch.Tensor, params: Dict):

        def ensure_2d(tensor):
            if len(tensor.shape) == 1:
                tensor = torch.unsqueeze(tensor, 0)
            return tensor
        
        state = ensure_2d(state)
        cmd = ensure_2d(cmd)

        # Unpack state tensor for readability
        # ---------------------------

        # this unbind method works on raw tensors, but not NM variables
        # x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r, wM1, wM2, wM3, wM4 = torch.unbind(state, dim=1)

        # try the state.unpack function that comes with the nm

        # this allows the NM to be compatible with 
        q0 =    state[:,3]
        q1 =    state[:,4]
        q2 =    state[:,5]
        q3 =    state[:,6]
        xdot =  state[:,7]
        ydot =  state[:,8]
        zdot =  state[:,9]
        p =     state[:,10]
        q =     state[:,11]
        r =     state[:,12]
        wM1 =   state[:,13]
        wM2 =   state[:,14]
        wM3 =   state[:,15]
        wM4 =   state[:,16]

        wMotor = torch.stack([wM1, wM2, wM3, wM4])
        wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])
        thrust = params["kTh"] * wMotor ** 2
        torque = params["kTo"] * wMotor ** 2

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = [torch.zeros(1, device=state.device)]*3

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = torch.stack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * torch.sign(velW * torch.cos(qW1) * torch.cos(qW2) - xdot)
                    * (velW * torch.cos(qW1) * torch.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    params["Cd"]
                    * torch.sign(velW * torch.sin(qW1) * torch.cos(qW2) - ydot)
                    * (velW * torch.sin(qW1) * torch.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    -params["Cd"] * torch.sign(velW * torch.sin(qW2) + zdot) * (velW * torch.sin(qW2) + zdot) ** 2
                    - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (params["IB"][1,1] - params["IB"][2,2]) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * params["dym"]
                )
                / params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (params["IB"][2,2] - params["IB"][0,0]) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * params["dxm"]
                )
                / params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((params["IB"][0,0] - params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / params["IB"][2,2],
            ]
        )

        ActuatorsDot = cmd/params["IRzz"]

        # State Derivative Vector
        # ---------------------------
        if state.shape[0] == 1:
            return torch.hstack([DynamicsDot.squeeze(), ActuatorsDot.squeeze()])
        else:
            return torch.hstack([DynamicsDot.T, ActuatorsDot])

    @staticmethod
    def neuromancer(state: torch.Tensor, cmd: torch.Tensor, params: Dict):

        # this allows the NM to be compatible with 
        q0 =    state[...,3]
        q1 =    state[...,4]
        q2 =    state[...,5]
        q3 =    state[...,6]
        xdot =  state[...,7]
        ydot =  state[...,8]
        zdot =  state[...,9]
        p =     state[...,10]
        q =     state[...,11]
        r =     state[...,12]
        wM1 =   state[...,13]
        wM2 =   state[...,14]
        wM3 =   state[...,15]
        wM4 =   state[...,16]

        wMotor = torch.stack([wM1, wM2, wM3, wM4])
        wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])
        thrust = params["kTh"] * wMotor ** 2
        torque = params["kTo"] * wMotor ** 2

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = [torch.zeros(1, device=state.device)]*3

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = torch.vstack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * torch.sign(velW * torch.cos(qW1) * torch.cos(qW2) - xdot)
                    * (velW * torch.cos(qW1) * torch.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    params["Cd"]
                    * torch.sign(velW * torch.sin(qW1) * torch.cos(qW2) - ydot)
                    * (velW * torch.sin(qW1) * torch.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    -params["Cd"] * torch.sign(velW * torch.sin(qW2) + zdot) * (velW * torch.sin(qW2) + zdot) ** 2
                    - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (params["IB"][1,1] - params["IB"][2,2]) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * params["dym"]
                )
                / params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (params["IB"][2,2] - params["IB"][0,0]) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * params["dxm"]
                )
                / params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((params["IB"][0,0] - params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / params["IB"][2,2],
            ]
        )

        # ensure the command is 2D
        if len(cmd.shape) == 3:
            cmd = cmd.squeeze(0)

        ActuatorsDot = cmd/params["IRzz"]
        return torch.hstack([DynamicsDot.T, ActuatorsDot])

class linmod:

    @staticmethod
    def pytorch(state: torch.Tensor, cmd: torch.Tensor, params: Dict):
        """
        This may look ridiculous, but it is imported from a symbolically derived linearisation in the 
        file jacobian_derivation.py
        """

        # Import State Vector
        # ---------------------------
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]
        xdot = state[7]
        ydot = state[8]
        zdot = state[9]
        p = state[10]
        q = state[11]
        r = state[12]
        wM1 = state[13]
        wM2 = state[14]
        wM3 = state[15]
        wM4 = state[16]

        wMotor = torch.stack([wM1, wM2, wM3, wM4])
        wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])

        # Stochastic Terms in Expectation
        # ---------------------------
        velW = ptu.from_numpy(np.array(0))
        qW1 =  ptu.from_numpy(np.array(0))
        qW2 =  ptu.from_numpy(np.array(0))

        # Jacobian A matrix
        # ---------------------------
        # placeholder for 0 which is a tensor to allow for easy stacking
        _ = ptu.from_numpy(np.array(0))

        # the contribution of x, y, z to the state derivatives
        col123 = ptu.from_numpy(np.zeros([17, 3]))

        # contribution of the attitude quaternion to the state derivatives
        col4567 = torch.vstack(
            [
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, -0.5 * p, -0.5 * q, -0.5 * r]),
                torch.stack([0.5 * p, _, 0.5 * r, -0.5 * q]),
                torch.stack([0.5 * q, -0.5 * r, _, 0.5 * p]),
                torch.stack([0.5 * r, 0.5 * q, -0.5 * p, _]),
                torch.stack(
                    [
                        -2
                        * q2
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q3
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q0
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q1
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        2
                        * q1
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        2
                        * q0
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q3
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q2
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        -2
                        * q0
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        2
                        * q1
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        2
                        * q2
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q3
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
            ]
        )

        # contribution of xdot, ydot, zdot to the state derivatives
        col8910 = torch.vstack(
            [
                torch.stack([_ + 1, _, _]),
                torch.stack([_, _ + 1, _]),
                torch.stack([_, _, _ + 1]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack(
                    [
                        (
                            params["Cd"]
                            * (-2 * velW * torch.cos(qW1) * torch.cos(qW2) + 2 * xdot)
                            * torch.sign(velW * torch.cos(qW1) * torch.cos(qW2) - xdot)
                        )
                        / params["mB"],
                        _,
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        (
                            params["Cd"]
                            * (-2 * velW * torch.sin(qW1) * torch.cos(qW2) + 2 * ydot)
                            * torch.sign(velW * torch.sin(qW1) * torch.cos(qW2) - ydot)
                        )
                        / params["mB"],
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        _,
                        (
                            -params["Cd"]
                            * (2 * velW * torch.sin(qW2) + 2 * zdot)
                            * torch.sign(velW * torch.sin(qW2) + zdot)
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
            ]
        )

        # contribution of p, q, r (body frame angular velocity) to the state derivatives
        cols11_12_13 = torch.vstack(
            [
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([-0.5 * q1, -0.5 * q2, -0.5 * q3]),
                torch.stack([0.5 * q0, -0.5 * q3, 0.5 * q2]),
                torch.stack([0.5 * q3, 0.5 * q0, -0.5 * q1]),
                torch.stack([-0.5 * q2, 0.5 * q1, 0.5 * q0]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack(
                    [
                        _,
                        (-params["IRzz"] * (wM1 - wM2 + wM3 - wM4) + r * (params["IB"][1,1] - params["IB"][2,2])) / params["IB"][0,0],
                        q * (params["IB"][1,1] - params["IB"][2,2]) / params["IB"][0,0],
                    ]
                ),
                torch.stack(
                    [
                        (params["IRzz"] * (wM1 - wM2 + wM3 - wM4) + r * (-params["IB"][0,0] + params["IB"][2,2])) / params["IB"][1,1],
                        _,
                        p * (-params["IB"][0,0] + params["IB"][2,2]) / params["IB"][1,1],
                    ]
                ),
                torch.stack([q * (params["IB"][0,0] - params["IB"][1,1]) / params["IB"][2,2], p * (params["IB"][0,0] - params["IB"][1,1]) / params["IB"][2,2], _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
            ]
        )

        # contribution of the angular accelerations of the rotors to the state derivatives
        cols_14_15_16_17 = torch.vstack(
            [
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack(
                    [
                        -2 * params["kTh"] * wM1 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                        -2 * params["kTh"] * wM2 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                        -2 * params["kTh"] * wM3 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                        -2 * params["kTh"] * wM4 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        2 * params["kTh"] * wM1 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                        2 * params["kTh"] * wM2 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                        2 * params["kTh"] * wM3 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                        2 * params["kTh"] * wM4 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        -2 * params["kTh"] * wM1 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                        -2 * params["kTh"] * wM2 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                        -2 * params["kTh"] * wM3 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                        -2 * params["kTh"] * wM4 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        (-params["IRzz"] * q + 2 * params["dym"] * params["kTh"] * wM1) / params["IB"][0,0],
                        (params["IRzz"] * q - 2 * params["dym"] * params["kTh"] * wM2) / params["IB"][0,0],
                        (-params["IRzz"] * q - 2 * params["dym"] * params["kTh"] * wM3) / params["IB"][0,0],
                        (params["IRzz"] * q + 2 * params["dym"] * params["kTh"] * wM4) / params["IB"][0,0],
                    ]
                ),
                torch.stack(
                    [
                        (params["IRzz"] * p + 2 * params["dxm"] * params["kTh"] * wM1) / params["IB"][1,1],
                        (-params["IRzz"] * p + 2 * params["dxm"] * params["kTh"] * wM2) / params["IB"][1,1],
                        (params["IRzz"] * p - 2 * params["dxm"] * params["kTh"] * wM3) / params["IB"][1,1],
                        (-params["IRzz"] * p - 2 * params["dxm"] * params["kTh"] * wM4) / params["IB"][1,1],
                    ]
                ),
                torch.stack(
                    [
                        -2 * params["kTo"] * wM1 / params["IB"][2,2],
                        2 * params["kTo"] * wM2 / params["IB"][2,2],
                        -2 * params["kTo"] * wM3 / params["IB"][2,2],
                        2 * params["kTo"] * wM4 / params["IB"][2,2],
                    ]
                ),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
            ]
        )

        A = torch.hstack([col123, col4567, col8910, cols11_12_13, cols_14_15_16_17])

        # Jacobian B matrix
        # ---------------------------

        # contribution of the input torques to the state derivatives
        B = torch.vstack(
            [
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_ + 1 / params["IRzz"], _, _, _]),
                torch.stack([_, _ + 1 / params["IRzz"], _, _]),
                torch.stack([_, _, _ + 1 / params["IRzz"], _]),
                torch.stack([_, _, _, _ + 1 / params["IRzz"]]),
            ]
        )

        return A, B

if __name__ == "__main__":

    # an example simulation and animation 
    from common.integrators import Euler, RK4
    from quadcopter.visualizer import Animator
    from parameters import get_quad_params

    ptu.init_dtype()
    ptu.init_gpu()

    def test_state_dot():
        quad_params = get_quad_params()
        state = quad_params["default_init_state_pt"]
        input = ptu.tensor([0.0001,0.0,0.0,0.0])

        Ti, Tf, Ts = 0.0, 3.0, 0.1
        memory = {'state': [ptu.to_numpy(state)], 'input': [ptu.to_numpy(input)]}
        times = np.arange(Ti, Tf, Ts)
        for t in times:

            state = RK4(state_dot.pytorch_vectorized, state, input, Ts, quad_params)

            memory['state'].append(ptu.to_numpy(state))
            memory['input'].append(ptu.to_numpy(input))

        memory['state'] = np.vstack(memory['state'])
        memory['input'] = np.vstack(memory['input'])

        animator = Animator(memory['state'], times, memory['state'], max_frames=10, save_path='data')
        animator.animate()

    # def test_mujoco_quad():
    #     quad_params = get_quad_params()
    #     state = quad_params["default_init_state_np"]
    #     input = np.array([0.0001,0.0,0.0,0.0])
    #     Ti, Tf, Ts = 0.0, 3.0, 0.1

    #     mj_quad = mujoco_quad(state=state, quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator='euler')
 
    #     memory = {'state': [state], 'input': [input]}
    #     times = np.arange(Ti, Tf, Ts)
    #     for t in times:
 
    #         state = mj_quad.step(input)
 
    #         memory['state'].append(state)
    #         memory['input'].append(input)
 
    #     memory['state'] = np.vstack(memory['state'])
    #     memory['input'] = np.vstack(memory['input'])
 
    #     # needs to be followed by a reset if we are to repeat the simulation
    #     mj_quad.reset(quad_params["default_init_state_np"])
 
    #     animator = Animator(memory['state'], times, memory['state'], max_frames=10, save_path='data')
    #     animator.animate()

    test_state_dot()

    print('fin')