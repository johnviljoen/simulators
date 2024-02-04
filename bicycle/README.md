# Bicycle Model

This will be a simple 2D top down view model. I will outline it here so that we know what to expect. For the following model:

$$
\begin{align} 
\dot{x}(t) &= v \cos(\psi) \\
\dot{y}(t) &= v \sin(\psi) \\
\dot{\psi}(t) &= \frac{v \tan(\phi)}{L}  \\
\dot{v}(t) &= a \\
\end{align}
$$

Where
- $a$ is the acceleration (m/s^2)
- $\phi$ is the steering angle (rad/s^2)
- $L$ is the vehicle wheel base = distance between front and rear wheels

The model is nonlinear due to the trigonometric terms cos, sin, and tan.

The control vector is {$a$, $\phi$}

If we are to linearise this system about a point, we would be able to extract a state space system from which to design control laws.

$$A = \begin{bmatrix}
0 & 0 & \cos(\psi) & -v\sin(\psi) \\
0 & 0 & \sin(\psi) & v\cos(\psi) \\
0 & 0 & \frac{\tan(\phi)}{L} & 0 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}$$

$$B = \begin{bmatrix}
0 & 0 \\
0 & 0 \\
0 & \frac{v \sec^2 (\phi)}{L} \\
1 & 0 \\
\end{bmatrix}$$

The final linear matrices at a given operating point will be found by substituting the values of the current states and inputs into the elements of the A and B matrices.

These matrices can then be converted to discrete time, either from first principles or using the scipy.signal.cont2discrete module.

This will provide us the linearised discrete time state space model for the car from which we can apply other control methods.