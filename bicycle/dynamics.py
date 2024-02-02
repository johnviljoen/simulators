import casadi as ca
import numpy as np

class KinematicBicycleModel:
    """
    x: {x, y, psi, v}
        0, 1, 2,   3  
    u: {a, phi}
        0, 1
    """
    def __init__(
            self, 
            wheelbase= 2.0, 
            max_steer= 30 * np.pi/180,
            backend = 'np_1d'
        ):

        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.backend = backend

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:

        xdot =  getattr(self, f"dynamics_{self.backend}")(x, u)
        return xdot

    def dynamics_np_1d(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:

        # input limit steering angle
        u[1] = np.clip(u[1], a_min=-self.max_steer, a_max=self.max_steer)

        # state wrap yaw from 0 to 2*pi
        # x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2]))

        return np.hstack([
            x[3]*np.cos(x[2]),
            x[3]*np.sin(x[2]),
            x[3]*np.tan(u[1]) / self.wheelbase,
            u[0],
        ])

    def dynamics_ca_1d(self, x: ca.SX, u: ca.SX) -> ca.SX:

        # state wrap yaw from 0 to 2*pi
        # x[2] = ca.arctan2(ca.sin(x[2]), ca.cos(x[2]))

        return ca.vertcat(
            x[3]*ca.cos(x[2]),
            x[3]*ca.sin(x[2]),
            x[3]*ca.tan(u[1]) / self.wheelbase,
            u[0],
        )
    
if __name__ == "__main__":
    Ts = 0.1
    dynamics = KinematicBicycleModel()
    x = np.array([0, 0, 0, 1.0])
    u = np.array([0.1, 30])
    xdot = dynamics(x, u)
    x_hist = []
    for i in range(1000):
        x += dynamics(x, u) * Ts
        x_hist.append(np.copy(x))
    x_hist = np.vstack(x_hist)
    import matplotlib.pyplot as plt
    plt.plot(x_hist[:,0], x_hist[:, 1])
    plt.show()
    print('fin')