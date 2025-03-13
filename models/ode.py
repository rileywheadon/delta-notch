# Import libraries
import numpy as np
import sys
import logging
import time
import numba
from scipy.integrate import solve_ivp

# Import configuration from config.py
from config import H_PLUS, H_MINUS, INITIAL_CELL, KT, G, GI, PI, PP, TIME, STEPS

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename = "main.log", filemode="w", level=logging.INFO)


# Perturb a variable with a sd of n * p
@numba.njit
def perturb(n, p):
    return n + np.random.normal(0, n * p)


# Define the ODE system
def ode_system(t, y, name, neighbours, size, mode):

    # Reshape into a matrix where each row represents a cell [N, D, I]
    state = y.reshape(size, 3)
    
    # Generate the derivative functions
    derivatives = np.empty((size, 3))
    for cell in range(size):

        # Get the neighobur indices
        mask = (neighbours[:, 0] == cell)
        indices = neighbours[mask, 1]

        # Compute the average delta signal over the neighbours
        divisor = 1
        if name == "Linear": divisor = 2
        if name == "Hexagonal": divisor = 6
        n_ext = np.sum(state[indices, 0]) / divisor
        d_ext = np.sum(state[indices, 1]) / divisor

        # Current cell values
        n, d, i = state[cell]
       
        # Use fixed equations in deterministic simulations
        dN = H_PLUS(i) - KT * n * d_ext - G * n
        dD = H_MINUS(i) - KT * d * n_ext - G * d
        dI = KT * n * d_ext - GI * i

        # Perturb parameter values in stochastic simulations
        if mode == "Stochastic":
            dN = H_PLUS(i) - perturb(KT, PP) * n * d_ext - perturb(G, PP) * n
            dD = H_MINUS(i) - perturb(KT, PP) * d * n_ext - perturb(G, PP) * d
            dI = perturb(KT, PP) * n * d_ext - perturb(GI, PP) * i

        derivatives[cell] = [dN, dD, dI]
    
    # Return flattened array
    return derivatives.flatten()


# Run a single simulation of the deterministic ODE model
def simulate(domain, mode):

    # Set the initial conditions
    name, neighbours, size = domain
    state = np.empty((size, 3))

    for i in range(size):
        for j in range(3):

            # Use a fixed initial state in stochastic simulations
            state[i][j] = INITIAL_CELL[j]

            # Perturb the initial state in deterministic simulations
            if mode == "Deterministic":
                state[i][j] = perturb(state[i][j], PI)

    # Return a solution to the system of differential equations
    res = solve_ivp(
        fun = ode_system, 
        t_span = (0, TIME), 
        y0 = state.flatten(), 
        t_eval = np.linspace(0, TIME, STEPS),
        args = (*domain, mode),
        rtol = 1e-6, 
        atol = 1e-9
    )

    return res.t, res.y.T.reshape(STEPS, size, 3)


# Run a gillespie model with given domain and size
def ode(domain, mode, samples = 1):
    
    # Initialize the data arrays
    data_time = []
    data_state = []

    # Run 'samples' simulations
    start = time.time()
    name, neighbours, size = domain
    logger.info(f"Model: {mode} ODE, Samples: {samples}, Domain: {name}")

    for i in range(samples):

        # In two cell simulations, order the cells 
        v_time, v_state = simulate(domain, mode)
        if size == 2 and v_state[-1, 1, 0] > v_state[-1, 0, 0]:
            v_state = np.roll(v_state, 1, axis = 1)

        # Add the time, state, and interpolated vectors to the data
        data_time.append(v_time)
        data_state.append(v_state)
        logger.info(f" - {time.time() - start:.4f}s: Finished simulation {i}")

    # Average over the interpolated data to get the mean
    data_mean = np.average(data_state, axis = 0) 
    logger.info(f"Finished in {time.time() - start:.4f}s")
    return data_time, data_state, data_mean



