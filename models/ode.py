# Import libraries
import numpy as np
import sys
import logging
import time
import numba
from scipy.integrate import solve_ivp

# Import configuration from config.py
from config import H_PLUS, H_MINUS, INITIAL_CELL, KT, G, GI, PI, NOISE, TIME, STEPS

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename = "main.log", filemode="w", level=logging.INFO)


# Define the ODE system
@numba.njit
def deterministic_step(domain, state):

    # Generate the increments 
    name, neighbours, size = domain
    derivatives = np.empty((size, 3))
    for cell in range(size):

        # Get the neighobur indices
        mask = (neighbours[:, 0] == cell)
        indices = neighbours[mask, 1]

        # Compute the average delta signal over the neighbours
        divisors = {"Two Cell": 1, "Linear": 2, "Hexagonal": 6}
        n_ext = np.sum(state[indices, 0]) / divisors[name]
        d_ext = np.sum(state[indices, 1]) / divisors[name]

        # Current cell values
        n, d, i = state[cell]
       
        # Use fixed equations in deterministic simulations
        dN = H_PLUS(i) - KT * n * d_ext - G * n
        dD = H_MINUS(i) - KT * d * n_ext - G * d
        dI = KT * n * d_ext - GI * i
        derivatives[cell] = [dN, dD, dI]
    
    # Return flattened array
    return derivatives


# Run a single simulation of the deterministic ODE model
@numba.njit
def simulate(domain, perturbations, wiener):

    # Initialize the states variable
    dt = TIME / STEPS
    name, neighbours, size = domain
    states = np.empty((STEPS, size, 3))

    # Set the initial condition 
    # Add perturbation (deterministic simulations only) 
    for i in range(size): 
        states[0, i] = INITIAL_CELL
        states[0, i] += perturbations[i]

    # Use a forward Euler method to simulate positive solutions to the ODE
    # Add wiener noise (stochastic simulations only)
    for i in range(1, STEPS):
        derivatives = deterministic_step(domain, states[i-1])
        states[i] = states[i - 1] + (derivatives * dt) 
        states[i] += states[i - 1] * NOISE * wiener[i]
        states[i] = np.maximum(0, states[i])

    return np.linspace(0, TIME, STEPS), states


# Return the initial perturbations and wiener values
def initialize_simulation(index, domain, mode, samples):

    # Default to perturbation and wiener coefficients of 0
    name, neighbours, size = domain
    perturbations = np.zeros((size, 3))
    wiener = np.zeros((STEPS, size, 3))

    # Generate consistent initial perturbations 
    if mode == "Deterministic-Linear" and size == 2:
        value = INITIAL_CELL * PI * ((index + 1) / samples)
        perturbations = np.array([value, -value])

    # Generate random initial perturbations 
    if mode == "Deterministic-Random":
        perturbations = np.random.normal(0, INITIAL_CELL * PI, (size, 3))

    # Generate wiener values 
    if mode == "Stochastic":
        dt = TIME / STEPS
        wiener = np.random.normal(0, np.sqrt(dt), (STEPS, size, 3))

    return perturbations, wiener


# Run a gillespie model with given domain and size
def ode(domain, mode, samples = 1):
    
    # Initialize the simulations
    name, neighbours, size = domain
    data_time, data_state, data_diff = [], [], []

    # Run simulations simulations
    start = time.time()
    logger.info(f"Model: {mode} ODE, Samples: {samples}, Domain: {name}")
    for i in range(samples):

        # Get the initial conditions and run the simulation
        perturbations, wiener = initialize_simulation(i, domain, mode, samples)
        v_time, v_state = simulate(domain, perturbations, wiener)

        # In two cell simulations, order the cells 
        if size == 2 and v_state[-1, 1, 0] > v_state[-1, 0, 0]:
            v_state = np.roll(v_state, 1, axis = 1)

        # Compute the differentiation time for two cell simulations
        if size == 2:
            max_index = np.argmax(v_state[:, 1, 0] < 1)
            data_diff.append(v_time[max_index])

        # Add the time, state, and interpolated vectors to the data
        data_time.append(v_time)
        data_state.append(v_state)
        logger.info(f" - {time.time() - start:.4f}s: Finished simulation {i}")

    # Get the mean and standard deviation of each sample
    data_mean = np.average(data_state, axis = 0)
    data_std = np.std(data_state, axis = 0)

    # Return the results of the simulation(s)
    logger.info(f"Finished in {time.time() - start:.4f}s")
    return data_time, data_state, data_mean, data_std, data_diff



