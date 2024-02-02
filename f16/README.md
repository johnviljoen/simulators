# F16_Gym

This is a differentiable PyTorch port of the nonlinear F16 simulation found [here](https://dept.aem.umn.edu/~balas/darpa_sec/SEC.Software.html#F16Manual). This allows for reinforcement learning algorithms reliant on a differentiable model.

Features:

- The dynamics model itself is found in dynamics.py, it utilises empirical lookup tables for simulating the F16 in a range of operating conditions given below.

- linearisation functions found in linmod.py, both numerically, and through using the PyTorch graph itself through autograd.functional.jacobian
- trimming function found in trim.py for the following flight regimes:
  - straight and level flight

## Dynamics

The lookup tables cover the following range of operating conditions:

### Full Simulation Flight Envelope

| parameter | minimum value | maximum value |  
| --------- | ------------- | ------------- |
| $\alpha$  | $-20 \deg$    | $90 \deg$     |  
| $\beta$   | $-30 \deg$    | $30 \deg$     |  
| $d_h$     | $-25 \deg$    | $25 \deg$     |  

NOTE: If using the Python lookup table implementation upon leaving the range of operating conditions the system is currently programmed to extrapolate using a zero order hold. In other words the value at an alpha of 90 degrees = value at 91 degrees = value at 500 degrees.

NOTE: If using the C lookup table implementation upon leaving the range of operating conditions the system will seg fault.

NOTE: the tables are not valid for compressible regimes, therefore their accuracy will break down above Mach 0.6. Further the minimum recommended velocity must be high enough such that the aircraft does not enter deep stall for high accuracy, therefore I have arbitrarily specified this as 200 ft/s in the table below.

NOTE: the tables are of a higher fidelity in an inner range specified in the table of recommended flight regimes below.

### Recommended Simulation Flight Envelope

| parameter | minimum value | maximum value |  
| --------- | ------------- | ------------- |
| $\alpha$  | $-10 \deg$    | $45 \deg$     |  
| $\beta$   | $-30 \deg$    | $30 \deg$     |  
| $d_h$     | $-25 \deg$    | $25 \deg$     |  
| $V$       | $200$ ft/s    | $675$ ft/s    |  

## Trimming

Naively utilising the Scipy.optimize.minimize Nelder-Mead technique, this trim function found in trim.py performs a trim of the aircraft in wings level steady flight at a given altitude and airspeed. This could be improved in future through use of the PyTorch graph itself.

### Testing

Performed in tests/integration_test_trim.py, the output trim is compared to three reference values found using the MATLAB implementation.

## Linearisation
