# Import libraries
import numpy as np
import sys
import logging
import time
import numba
from scipy.interpolate import interp1d

# Import modules
from config import H_PLUS, H_MINUS, DEFAULT, TIME, STEPS


# Generates the reaction rates based on the state
@numba.njit
def update_reaction_rates(state, neighbours, params):
    K, NM, DM, N0, D0, KT, G, GI = params

    # Initialize empty rate arrays for each reaction
    r1 = np.empty(len(neighbours))
    r2 = np.empty(len(state))
    r3 = np.empty(len(state))
    r4 = np.empty(len(state))
    r5 = np.empty(len(state))
    r6 = np.empty(len(state))

    # Iterate over all cells to get the reaction/decay rates
    for i, cell in enumerate(state):

        # Set all of the decay rates (reactions 2-4)
        r2[i] = cell[0] * G  # Rate parameter of Notch decay
        r3[i] = cell[1] * G  # Rate parameter of Delta decay
        r4[i] = cell[2] * GI  # Rate parameter of NICD decay

        # Set all of the production rates (reactions 5-6)
        r5[i] = H_PLUS(params, cell[2])
        r6[i] = H_MINUS(params, cell[2])

    # Sum over the neighbour pairs to get the binding rates (reaction 1) 
    for i, (cellA, cellB) in enumerate(neighbours):
        r1[i] = KT * state[cellA][0] * state[cellB][1]

    # Return the combined array of rates
    return np.concatenate((r1 ,r2, r3, r4, r5, r6))


# Takes the reaction rates and generates the next reaction time
@numba.njit
def generate_reaction_time(rates):
    return np.random.exponential(scale = 1 / np.sum(rates))


# Generate an event, returning a (reaction, index) tuple
@numba.njit
def generate_event(rates, state, neighbours):

    # Sample a random value in [0, 1] and find its index in the partition
    partition = np.cumsum(rates) / np.sum(rates)
    reaction = np.searchsorted(partition, np.random.rand())

    # Check if the event is a binding event
    if reaction < len(neighbours):
        return [1, reaction]

    # Otherwise do some modulo fuckery to get the reaction
    reaction_type = 2 + ((reaction - len(neighbours)) // len(state))
    index = (reaction - len(neighbours)) % len(state)
    return [reaction_type, index]


# Updates the state AND the reaciton rates given an event
@numba.njit
def update_state(state, event, neighbours):
    reaction, index = event

    # Binding reaction event
    if reaction == 1:
        cellA, cellB = neighbours[index]
        state[cellA][0] -= 1 # Remove Notch from cell A
        state[cellB][1] -= 1 # Remove Delta from cell B
        state[cellA][2] += 1 # Add a NICD to cell A

    # Notch decay event
    if reaction == 2: 
        state[index][0] -= 1

    # Delta decay event
    if reaction == 3: 
        state[index][1] -= 1

    # NICD decay devent
    if reaction == 4:
        state[index][2] -= 1

    # Notch production event
    if reaction == 5: 
        state[index][0] += 1

    # Delta production event
    if reaction == 6:
        state[index][1] += 1

    return state
 

# Runs a single simulation 
@numba.njit
def gillespie(domain, initial, params = DEFAULT, stop = 10 ** 5):

    # Initialize the simulation
    t = 0
    state = initial
    name, neighbours, size = domain
    rates = np.empty(len(neighbours) + (size * 5))

    # Initialize the state and time vectors
    vT, vS = [], []
    while t <= TIME and len(vT) < stop:

        # Update reaction rates
        rates = update_reaction_rates(state, neighbours, params)

        # Add the current time to the time vector
        vT.append(t)

        # Add the state to the state vector
        vS.append(np.copy(state))

        # Generate a new reaction time and increment time
        t += generate_reaction_time(rates)

        # Generate an event for this time step
        event = generate_event(rates, state, neighbours)

        # Update the state based on the event
        state = update_state(state, event, neighbours)

    return vT, vS

