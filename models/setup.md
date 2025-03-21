# Delta-Notch Computing Environment

This is an computing environment for testing three different models of Delta-Notch signalling:

1. A deterministic ODE model.
2. An ODE model with additional stochastic terms.
3. An agent-based stochastic model.

These models can be run on three different types of domains: two-cell, linear, and hexagonal. The linear and hexagonal domains may use an arbitrary number of cells. They can also use either Dirichlet (zero-flux) or periodic boundary conditions (the boundary of the two-cell system is always periodic).

Additionally, each simulation can be fine-tuned using a list of parameters:

- `K`: The hill function coefficient for Delta and Notch production.
- `NM`: The maximum rate of Notch production.
- `DM`: The maximum rate of Delta production.
- `N0`: The hill function half-max for Notch production.
- `D0`: The hill function half-max for Delta production.
- `KT`: The Delta-Notch binding rate.
- `G`: The rate of Delta and Notch decay.
- `GI`: The rate of NICD (Notch Intracellular Domain) decay.

## Setup

The `ode.py` and `gillespie.py` files contain compiled Python code for rapidly simulating all three of the models above. They require a domain object (see `domains.py`) and set of parameter values. They return the following two objects:

- `vT`: A vector of times. For the ODE models, these are the evenly-spaced points used to compute the Forward Euler approximation of the solution. For the agent-based model, `vT` is the vector of times at which events occurred, which _will not_ be evenly-spaced.
- `vS`: A vector of states at each time step. It is a three dimensional array where the first dimension is the time step, the second dimension is the cell, and the third dimension is the molecule count for Notch, Delta, and NICD (in that order).

The `helpers.py` file contains various helper functions for computing relevant statistics:

- `interpolate(p, vT, vS)` interpolates `vT` and `vS` at `p` evenly spaced points, returning a new time vector and state vector. This improves plot legibility and is useful for comparing the results from the ODE and agent-based models, which have different time vectors.
- `summarize(states)` computes the mean and standard deviation of a list of state vectors. This function only works if all state vectors have the same size, so make sure to use `interpolate` first if necessary.
- `differentiation(vT, vS)` computes the _differentiation time_ of the given time and state vector, defined as the point at which all cells have reached their final fate.
  - If the model failed to differentiate, this function returns $0$.
  - Ensure simulations are run for a sufficiently long time, or this function won't work.
- `pattern(vS)` returns the _pattern_ created by given simulation, returned as an array of binary values ($0$ for zero Notch, $1$ nonzero Notch) with the same length as the number of cells in the domain.

The `experiments.py` file contains code for running specific experiments described below. The `visualization.py` file contains visualization code for each experiment. The `main.py` file runs all experiments by default, but one should comment out unnecessary experiments to reduce compute time.

## Experiments

Each experiment has a unique numerical ID, which makes them easier to identify.

### Basic

- `01`: 10 simulations of the deterministic model (two-cell domain).
- `02`: 10 simulations of the stochastic model (two-cell domain).
- `03`: 10 simulations of the agent-based model (two-cell domain).
- `04`: 1 simulation of the deterministic model (7-cell linear dirichlet domain).
- `05`: 1 simulation of the stochastic model (7-cell linear dirichlet domain).
- `06`: 1 simulation of the agent-based model (7-cell linear dirichlet domain).
- `07`: 1 simulation of the deterministic model (9-cell hexagonal periodic domain).
- `08`: 1 simulation of the stochastic model (9-cell hexagonal periodic domain).
- `09`: 1 simulation of the agent-based model (9-cell hexagonal periodic domain).

### Two-Cell

- `10`: How does varying the initial perturbation affect differentiation time in the deterministic two-cell model? To do this, we run simulations and plot perturbation size against differentiation time.
- `11`: How does noise affect differentiation time in the stochastic two-cell model? Which noise coefficient most accurately replicates the behaviour of the Gillespie model? Use a KDE to estimate the distribution.
- `12`:
- `13`: How does changing the parameters (particularly `NM`, `DM`, `KT`, `G`, `GI`) affect whether the cells differentiate or not? Search the parameter space for all three types of models.

### Linear

- `14`: How do odd/even domains and dirichlet/periodic boundary conditions affect cell patterns? Test all combinations and make a plot showing the different patterns and their frequency. Use the Gillespie model.
- `15`: How do dirichlet/periodic boundary conditions affect differentiation times? Use the Gillespie model.
- `16`: Redo experiment `12` on a linear domain. Search the parameter space for all three types of models.

### Hexagonal

- `17`: How does domain shape affect differentiation time and the number of cells with each fate? Use the Gillespie model. Test wide and flat as well as more regular domains.
- `18`: How do dirichlet/periodic boundary conditions affect cell patterns? Use a 5x5 domain and the Gillespie model. Show each pattern (or just the top 5 if there's a lot of them) and their frequency.
