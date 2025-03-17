# Import libraries
import numpy as np
import sys
import logging
import time
import numba
from scipy.interpolate import interp1d

# Import configuration from config.py
from config import H_PLUS, H_MINUS, INITIAL_CELL, KT, G, GI, TIME, STEPS

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename = "main.log", filemode="w", level=logging.INFO)


# Generates the reaction rates based on the state
@numba.njit
def update_reaction_rates(state, neighbours):

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
        r5[i] = H_PLUS(cell[2])   # Rate parameter of Notch production
        r6[i] = H_MINUS(cell[2])   # Rate parameter of Delta production

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
def simulate(domain):

    # Initialize the simulation
    t = 0
    name, neighbours, size = domain
    state = np.empty((size, 3))
    rates = np.empty(len(neighbours) + (size * 5))

    # Unfortunate numba hack to set the state vector
    for i in range(size): 
        state[i] = INITIAL_CELL

    # Initialize the state and time vectors
    v_time = []
    v_state = []
    while t <= TIME:

        # Update reaction rates
        rates = update_reaction_rates(state, neighbours)

        # Add the current time to the time vector
        v_time.append(t)

        # Generate a new reaction time and increment time
        t += generate_reaction_time(rates)

        # Generate an event for this time step
        event = generate_event(rates, state, neighbours)

        # Update the state based on the event
        state = update_state(state, event, neighbours)

        # Add the state to the state vector
        v_state.append(np.copy(state))

    # Append this sample to the data vectors
    return v_time, v_state


# Run a gillespie model with given domain and size
def gillespie(domain, samples = 1):
    
    # Initialize the data arrays
    data_time = []
    data_state = []
    data_mean = []
    data_diff = []

    # Run 'samples' simulations
    start = time.time()
    name, neighbours, size = domain
    logger.info(f"Model: Gillespie, Samples: {samples}, Domain: {name}")

    for i in range(samples):

        v_time, v_state = simulate(domain)
        v_time = np.array(v_time)
        v_state = np.array(v_state)

        # In two cell simulations, order the cells 
        if size == 2 and v_state[-1, 1, 0] > v_state[-1, 0, 0]:
            v_state = np.roll(v_state, 1, axis = 1)

        # Compute the differentiation time for two cell simulations
        if size == 2:
            max_index = np.argmax(v_state[:, 1, 0] < 1)
            data_diff.append(v_time[max_index])

        # Linearly interpolate the state vector
        f_interp = interp1d(v_time, v_state, axis = 0, fill_value = "extrapolate")
        v_interp = f_interp(np.linspace(0, TIME, STEPS))

        # Add the time, state, and interpolated vectors to the data
        data_time.append(v_time)
        data_state.append(v_state)
        data_mean.append(v_interp)
        logger.info(f" - {time.time() - start:.4f}s: Finished simulation {i}")

    # Average over the interpolated data to get STD and the mean
    data_std = np.std(data_mean, axis = 0)
    data_mean = np.average(data_mean, axis = 0) 
    logger.info(f"Finished in {time.time() - start:.4f}s")
    return data_time, data_state, data_mean, data_std, data_diff



