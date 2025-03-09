# IMPORT PYTHON MODULES
import matplotlib.pyplot as plt
import numpy as np
import logging

# INITIALIZE PARAMETER VALUES
KT = 0.0001  # Delta-Notch binding rate
GE = 0.002   # Extracellular molecule decay rate (used in dN/dt and dD/dt equations)
GI = 0.01    # NICD decay rate
NM = 5       # Maximum rate of Notch production
DM = 5       # Maximum rate of Delta production
N0 = 100     # Notch hill function (H+) half-max
D0 = 100     # Delta hill function (H-) half-max
N = 2        # Hill function parameter

# INITIALIZE HILL FUNCTIONS
HP = lambda i : (NM * i**N) / ((N0**N) + (i**N))
HM = lambda i : (DM * D0**N) / ((D0**N) + (i**N))

# INITIALIZE NEIGHBOURS (by index in state, include both directions)
NEIGHBOURS = np.array([
    [0, 1],
    [1, 0],
])

# INITIALIZE SYSTEM STATE (each element contains integers [N, D, I] for one cell)
state = np.array([
    [500, 50, 200],
    [500, 50, 200],
])

# INITIALIZE SYSTEM RATES (ri contains rates for reaction i)
#  - Each pair of neighbours creates one reaction
#  - Each cell has five other reactions (decay/production)
rates = np.empty(len(NEIGHBOURS) + (len(state) * 5))

# INITIALIZE SIMULATION TIME
t = 0


# Generates the reaction rates based on the current state
def update_reaction_rates(state):

    # Initialize empty rate arrays for each reaction
    r1 = np.empty(len(NEIGHBOURS))
    r2 = np.empty(len(state))
    r3 = np.empty(len(state))
    r4 = np.empty(len(state))
    r5 = np.empty(len(state))
    r6 = np.empty(len(state))

    # Iterate over all cells to get the reaction/decay rates
    for i, cell in enumerate(state):

        # Set all of the decay rates (reactions 2-4)
        r2[i] = cell[0] * GE  # Rate parameter of Notch decay
        r3[i] = cell[1] * GE  # Rate parameter of Delta decay
        r4[i] = cell[2] * GI  # Rate parameter of NICD decay

        # Set all of the production rates (reactions 5-6)
        r5[i] = HP(cell[2])   # Rate parameter of Notch production
        r6[i] = HM(cell[2])   # Rate parameter of Delta production

    # Sum over the neighbour pairs to get the binding rates (reaction 1) 
    for i, (cellA, cellB) in enumerate(NEIGHBOURS):
        r1[i] = KT * state[cellA][0] * state[cellB][1]

    # Return the combined array of rates
    return np.concatenate((r1 ,r2, r3, r4, r5, r6))


# Takes the reaction rates and generates the next reaction time
def generate_reaction_time(rates):
    return np.random.exponential(scale = np.sum(rates))


# Generate an event, returning a (reaction, index) tuple
def generate_event(rates):

    # Sample a random value in [0, 1] and find its index in the partition
    partition = np.cumsum(rates) / np.sum(rates)
    event = np.searchsorted(partition, np.random.rand())

    # Check if the event is a binding event
    if event < len(NEIGHBOURS):
        return (np.int64(1), event)

    # Otherwise do some modulo fuckery to get the reaction
    reaction = 2 + ((event - len(NEIGHBOURS)) // len(state))
    index = (event - len(NEIGHBOURS)) % len(state)
    return (reaction, index)


# Update the state given an event
def update_state(event):
    reaction, index = event

    # Binding reaction event
    if reaction == 1:
        cellA, cellB = NEIGHBOURS[index]
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
    

# Main simulation loop
def main():

    global state
    global rates
    global t

    # Initialize an empty results array
    steps = 50000
    vT = np.arange(steps)
    results = np.empty((steps, 2, 3))
    for i in vT:

        # Update reaction rates
        rates = update_reaction_rates(state)

        # Generate a new reaction time and increment t
        t += generate_reaction_time(rates)

        # Generate an event for this time step
        event = generate_event(rates)

        # Update the state based on the event
        state = update_state(event)

        # Add the state to the results array
        results[i] = state

    # Plot the results
    fig, axs = plt.subplots(3, 1, sharex = True)
    axs[0].set_ylabel("NOTCH")
    axs[0].plot(vT, results[:, 0, 0])
    axs[0].plot(vT, results[:, 1, 0])
    axs[1].set_ylabel("DELTA")
    axs[1].plot(vT, results[:, 0, 1])
    axs[1].plot(vT, results[:, 1, 1])
    axs[2].set_ylabel("NICD")
    axs[2].plot(vT, results[:, 0, 2])
    axs[2].plot(vT, results[:, 1, 2])
    # plt.show()
    fig.savefig('fig.pdf')

main()
