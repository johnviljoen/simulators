# Tests

As I wrote this code, I found myself writing a lot of code to test its various functionality.
I decided to keep this code and turn them into unit/integration/endtoend tests instead of
deleting it.

It is my hope that this can be a record of my thought process, and is sufficient to show others
that I have been rigorous in the testing and development of this model.

The MATLAB_timehistory directory contains some saved MATLAB trajectories of the original model.
I have only used two of them in these tests so far, '10000ft_700fts_x_sim' and the 'lef_dist...'
files as this was sufficient for me to be satisfied that the model is accurate to the MATLAB.
Note that all of these reference timehistories start at a MATLAB trimmed state of wings level
cruise.

The largest discrepancy that I am leaving with is the fact that Simulink, despite being told to
use a timestep of 0.1 for the LEF actuator, decided to use two different smaller timesteps for
both of the LEF states. I am sure I am at fault as I would be shocked if I discovered a bug in
Simulink itself, but that is the only non-rigorously tested part of this simulation in my opinion,
although I am open to be proved wrong!
