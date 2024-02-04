import os
import copy

import numpy as np
import mujoco as mj
import quadcopter.mujoco.utils as utils

class Interface:

    """
    The rather annoying thing about mujoco is it is stateful, it holds a state
    within it, which I must get and set and reset when necessary. That being said
    I will not hold an internal memory in this class of the state on the python side
    for simplicity.

    I do however need to hold a state of the omegas, as mujoco does not simulate them.
    I wish it did not have to be like this.
    """

    def __init__(
            self,
            state,
            quad_params,
            Ti, Tf, Ts,
            integrator,
            xml_path='dynamics.xml',
            render='matplotlib' # 'matplotlib', 'mujoco'    
        ):

        self.integrator = integrator
        self.Ts = Ts
        self.Ti = Ti
        self.Tf = Tf
        self.quad_params = quad_params # used to do a lot of the instantiating
        self.t = Ti
        self.render = render
        self.xml_path = xml_path

        self.omegas = copy.deepcopy(state[13:17])

        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path, )  # MuJoCo model
        self.model.opt.timestep = self.Ts

        assert self.Ts == self.model.opt.timestep # should be 0.01
        # assert self.model.opt.integrator == 0 # 0 == semi implicit euler, closest to explicit euler mj offers
        # mjdata constains the state and quantities that depend on it.
        if self.integrator == 'euler':
            # this is NOT explicit euler this is semi-implicit euler
            self.model.opt.integrator = 0
        elif self.integrator == 'rk4':
            self.model.opt.integrator = 1
        self.data = mj.MjData(self.model)

        # Make renderer, render and show the pixels
        if render == 'mujoco':
            self.renderer = mj.Renderer(model=self.model, height=720, width=1280)
            # self.data.cam_xpos = np.array([[1,2,3]])
            self.model.cam_pos0 = np.array([[1,2,3]])
            self.model.cam_pos = np.array([[1,2,3]])

        mj.mj_resetData(self.model, self.data)  # Reset state and time.
        self.data.ctrl = [self.quad_params["kTh"] * self.quad_params["w_hover"] ** 2] * 4 # kTh * w_hover ** 2 = 2.943

        self.current_cmd = np.zeros(4)

        # set initial conditions, track with state attribute for convenience
        self.set_state(copy.deepcopy(state))

        # mujoco operates on numpy arrays not tensors
        self.state = copy.deepcopy(state)

    def __call__(self, state: np.ndarray, cmd: np.ndarray):
        return self.step(cmd.squeeze(0))

    def step(
            self, 
            cmd: np.ndarray,
        ):
        assert isinstance(cmd, np.ndarray), "cmd should be a np.ndarray for mujoco sim"
        self.current_cmd = cmd

        # translate omegas to thrust (mj input)
        thr = self.quad_params["kTh"] * self.omegas ** 2
        self.data.ctrl = thr.tolist()

        # update mujoco and actuators with EULER
        mj.mj_step(self.model, self.data)
        self.omegas += cmd/self.quad_params["IRzz"] * self.Ts

        # retrieve time for the environment
        self.t = self.data.time

        self.state = self.get_state()

        return self.state

    def get_state(self):

        return utils.mj_get_state(self.data, self.omegas)

    def set_state(self, state):

        # convert state to mujoco compatible 
        qpos, qvel = utils.state2qpv(state)

        # apply
        self.data.qpos = qpos
        self.data.qvel = qvel

        # handle the rotors (omegas) and state save separately
        self.omegas = copy.deepcopy(state.squeeze()[13:17])
        self.state = copy.deepcopy(state)

    def reset(self, state):

        print('performing mujoco reset')

        self.omegas = np.array([self.quad_params["w_hover"]]*4)
        self.set_state(copy.deepcopy(state))

        # added during sysID phase
        self.t = self.Ti
    
if __name__ == "__main__":

    # an example simulation and animation 
    from common.integrators import Euler, RK4
    from common import pytorch_utils as ptu
    from quadcopter.visualizer import Animator
    from quadcopter.parameters import get_quad_params

    ptu.init_dtype()
    ptu.init_gpu()

    def test_mujoco_quad():
        quad_params = get_quad_params()
        state = quad_params["default_init_state_np"]
        input = np.array([0.0001,0.0,0.0,0.0])
        Ti, Tf, Ts = 0.0, 3.0, 0.1
        mj_quad = Interface(state=state, quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator='euler')
        memory = {'state': [state], 'input': [input]}
        times = np.arange(Ti, Tf, Ts)
        for t in times:
            state = mj_quad.step(input)
            memory['state'].append(state)
            memory['input'].append(input)
        memory['state'] = np.vstack(memory['state'])
        memory['input'] = np.vstack(memory['input'])
        # needs to be followed by a reset if we are to repeat the simulation
        mj_quad.reset(quad_params["default_init_state_np"])
        animator = Animator(memory['state'], times, memory['state'], max_frames=10, save_path='data')
        animator.animate()

    test_mujoco_quad()
