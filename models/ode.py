# Import libraries
import numpy as np
import sys
import logging
import time
import numba
from scipy.integrate import solve_ivp


# Import modules
from helpers import *
from config import H_PLUS, H_MINUS, DEFAULT, TIME, STEPS


# Define the ODE system
@numba.njit
def deterministic_step(domain, state, params):

    # Unpack parameter and domain information
    K, NM, DM, N0, D0, KT, G, GI = params
    name, neighbours, size = domain

    # Generate the increments 
    increments = np.empty((size, 3))
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
        dN = H_PLUS(params, i) - KT * n * d_ext - G * n
        dD = H_MINUS(params, i) - KT * d * n_ext - G * d
        dI = KT * n * d_ext - GI * i
        increments[cell] = [dN, dD, dI]
    
    # Return flattened array
    return increments


# Run an ODE model (noise = 0 for deterministic)
@numba.njit
def ode(domain, initial, noise = 0, params = DEFAULT, time = TIME, steps = STEPS):

    # Compute the wiener values
    dt = time / steps
    name, neighbours, size = domain
    wiener = noise * np.random.normal(0, np.sqrt(dt), (steps, size, 3))
    
    # Initialize the simulation
    states = np.empty((steps, size, 3))
    states[0] = initial

    # Use a forward Euler method to simulate positive solutions to the ODE
    for i in range(1, steps):
        increments = deterministic_step(domain, states[i-1], params)
        states[i] = states[i - 1] + (increments * dt) 
        states[i] += states[i - 1] * wiener[i]
        states[i] = np.maximum(0, states[i])

    return np.linspace(0, time, steps), states



