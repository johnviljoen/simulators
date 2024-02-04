# Overview

Each directory contains the files specific to its respective simulator. Common code amongst different simulators is defined in the "common" directory. Each simulator is independent of any other (with the exception that the mujoco quadcopter simulator relies on the base quadctoper simulator), and some require different dependencies. All are installed by the setup.py pip install -e .

# Simulators

## Bicycle - working

This is a standard planar bicycle model

- $ state = \{ x, y, \theta, \psi \}$
- $ input = \{ a, \dot{\psi} \} $

## Quadcopter - working

- $ state = \{  \} $
- $ input = \{  \} $

### Mujoco Backend - working

This is a Mujoco copy of the system dynamics, relies on the user having a local install of mujoco: (link installer)

## F16 - working

- $ state = \{ x, y, z, \phi, \theta, \psi,  V, \alpha, \beta, p, q, r, T, \delta_h,  \delta_a,  \delta_r,  \delta_{LEF}, LF\} $
- $ input = \{ T, \delta_h,  \delta_a,  \delta_r \} $

NOTE: the input represents the DESIRED version of those states, which are then used as references in P controllers to reach those desired states. Also it is a shame that the EoM are not setup to use quaternions here - a future improvement I am sure.

$$
\begin{align}
\dot{x} &= U(\cos\theta \cos\psi) + V(\sin\phi \cos\psi \sin\theta - \cos\phi \sin\psi) + W(\cos\phi \sin\theta \cos\psi + \sin\phi \sin\psi) \\
\dot{y} &= U(\cos\theta \sin\psi) + V(\sin\phi \sin\psi \sin\theta + \cos\phi \cos\psi) + W(\cos\phi \sin\theta \sin\psi - \sin\phi \cos\psi) \\
\dot{z} &= U \sin\theta - V(\sin\phi \cos\theta) - W(\cos\phi \cos\theta) \\
\dot{\phi} &= P + \tan\theta(Q \sin\phi + R \cos\phi) \\
\dot{\theta} &= Q \cos\phi - R \sin\phi \\
\dot{\psi} &= \frac{Q \sin\phi + R \cos\phi}{\cos\theta} \\
\dot{vt} &= \frac{U\dot{U} + V\dot{V} + W\dot{W}}{vt} \\
\dot{\alpha} &= \frac{U\dot{W} - W\dot{U}}{U^2 + W^2} \\
\dot{\beta} &= \frac{\dot{V}v_t - V\dot{v}_t}{v_t^2 \cos\beta} \\
\dot{P} &= \frac{J_z L_{\text{tot}} + J_{xz} N_{\text{tot}} - (J_z(J_z-J_y)+J_{xz}^2)QR + J_{xz}(J_x-J_y+J_z)PQ + J_{xz}QH_{\text{eng}}}{J_x J_z - J_{xz}^2} \\
\dot{Q} &= \frac{M_{\text{tot}} + (J_z-J_x)PR - J_{xz}(P^2-R^2) - RH_{\text{eng}}}{J_y} \\
\dot{R} &= \frac{J_x N_{\text{tot}} + J_{xz} L_{\text{tot}} + (J_x(J_x-J_y)+J_{xz}^2)PQ - J_{xz}(J_x-J_y+J_z)QR + J_x QH_{\text{eng}}}{J_x J_z - J_{xz}^2} \\
\end{align}
$$

Complete equations need to be finished - to fully explain the L,M,N total forces determined by the aerodynamic data tables.

## Satellite - working

- $ state = \{  \} $
- $ input = \{  \} $

# Feature Matrix

| Simulation | Quaternion Attitude | Linearisation | Animation | Differentiable |
| ---------- | ------------------- | ------------- | --------- | -------------- |
| Bicycle    | -                   | Y             | N         | Y
| F16        | N                   | Y             | N         | Y
| Quadcopter | Y                   | Y             | Y         | Y
| Satellite  | Y                   | N             | Y         | Y
| Motorcycle | -                   | -             | -         | -