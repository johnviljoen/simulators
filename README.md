# Overview

Each directory contains the files specific to its respective simulator. Common code amongst different simulators is defined in the "common" directory. Each simulator is independent of any other (with the exception that the mujoco quadcopter simulator relies on the base quadctoper simulator), and some require different dependencies. All are installed by the setup.py pip install -e .

# Simulators

## Bicycle

This is a standard planar bicycle model

- $ state = \{ x, y, \theta, \psi \}$
- $ input = \{ a, \dot{\psi} \} $

## Quadcopter - working

- $ state = \{  \} $
- $ input = \{  \} $

### Mujoco Backend - working

This is a Mujoco copy of the system dynamics, relies on the user having a local install of mujoco: (link installer)

## F16

- $ state = \{  \} $
- $ input = \{  \} $

## Satellite - working

- $ state = \{  \} $
- $ input = \{  \} $


