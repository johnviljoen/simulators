import torch
import casadi as ca

import common.pytorch_utils as ptu
from common.rotation import quaternion_derivative

class state_dot:

    """
    Compute the time derivative of the state. This is the J1 dynamics alone.

    Parameters:
    - state: Tensor containing the state.               [minibatch, nx]
            {x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r}
             0  1  2  3   4   5   6   7     8     9     10 11 12
    - action: Tensor containing the control inputs.     [minibatch, nu]
            {x2dot, y2dot, z2dot, pdot, qdot, rdot}
             0      1      2      3     4     5
    - G: Gravitational constant.
    - M: Mass of the Earth.
    
    Returns:
    - Tensor containing the time derivative of the state concatenated with the orbital elements
            {[state_dot]}
              0 : 13
    """

    @staticmethod
    def pytorch_batched(state: torch.Tensor, action: torch.Tensor, G=6.67430e-11, M=5.972e+24):

        # Extract positions and velocities
        pos = state[:, :3]  # [minibatch, 3]
        vel = state[:, 7:10]  # [minibatch, 3]

        # Compute the square of the distances
        r_squared = torch.sum(pos * pos, dim=1, keepdim=True)  # [minibatch, 1]

        # Compute the gravitational acceleration
        r = torch.sqrt(r_squared)                   # [minibatch, 1]
        acc_gravity = -G * M / r_squared * pos / r  # [minibatch, 3]

        # Quaternion derivatives need to be calculated with respect to angular rates
        # This quaternion derivative will assume q = [q0, q1, q2, q3] where q0 is the scalar part
        q = state[:, 3:7]  # Extract quaternions from the state
        omega = state[:, 10:13]  # Extract angular velocities (p, q, r)

        q_dot = quaternion_derivative.pytorch_batched(q, omega)

        # Compute the time derivative of the state
        state_dot = torch.hstack([
            vel,
            q_dot,
            acc_gravity + action[:, :3],
            action[:, 3:]
        ])

        return state_dot
    
    @staticmethod
    def pytorch(state, action):
        raise NotImplementedError
    
    @staticmethod
    def casadi(state: ca.MX, action: ca.MX, G=6.67430e-11, M=5.972e+24):
        # state is a column vector [nx, 1] where nx is the size of the state
        # action is a column vector [nu, 1] where nu is the size of the action
        
        # Extract positions and velocities
        pos = state[:3]  # [3, 1]
        vel = state[7:10]  # [3, 1]

        # Compute the square of the distances
        r_squared = ca.sumsqr(pos)  # scalar

        # Compute the gravitational acceleration
        r = ca.sqrt(r_squared)  # scalar
        acc_gravity = -G * M / r_squared * pos / r  # [3, 1]

        # Extract quaternions and angular velocities
        q = state[3:7]  # [4, 1]
        omega = state[10:13]  # [3, 1]

        # Here you would call your quaternion_derivative function, adapted for CasADi
        # Assuming that a casadi_quaternion_derivative function exists
        q_dot = quaternion_derivative.casadi(q, omega)

        # Concatenate to form the time derivative of the state
        state_dot = ca.vertcat(
            vel,
            q_dot,
            acc_gravity + action[:3],
            action[3:]
        )

        return state_dot
    
    @staticmethod
    def numpy_batched(state, action):
        raise NotImplementedError
    
    @staticmethod
    def numpy(state, action):
        raise NotImplementedError
    
if __name__ == "__main__":

    import torch
    from parameters import get_params
    from utils import state_to_orbital_elements
    import numpy as np
    from visualizer import Animator

    # example usage
    ptu.init_gpu(use_gpu=True, gpu_id=0)
    ptu.init_dtype(set_dtype=torch.float32)

    # setup
    # -----

    params = get_params()
    Ti, Ts, Tf = 0, 10.0, 10000
    state = torch.clone(params["init_state"])
    state[:,8] += 1000
    state[:,9] += 100
    action = ptu.tensor([[0.,0,0,0,0,0]])

    # simulate
    # --------

    state_history = [ptu.to_numpy(state)]
    times = torch.arange(Ti, Tf, Ts)
    for t in times:
        state += state_dot.pytorch_batched(state, action) * Ts
        print(state_to_orbital_elements.pytorch_batched(state))
        state_history.append(ptu.to_numpy(state))
        print(t)

    # animate 
    # -------

    state_history = np.vstack(state_history)
    reference_history = np.copy(state_history)

    animator = Animator(
        states=state_history[::10,:], 
        references=reference_history[::10,:], 
        times=ptu.to_numpy(times[::10]),
        sphere_rad=5000e3
    )
    animator.animate()

    print('fin')