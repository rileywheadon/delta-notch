import numpy as np
import numba

# Parameter values
K = 2        # Hill function exponent
NM = 10      # Maximum rate of Notch production
DM = 10      # Maximum rate of Delta production
N0 = 100     # Notch hill function (H+) half-max
D0 = 100     # Delta hill function (H-) half-max
KT = 0.0001  # Delta-Notch binding rate
G = 0.02     # Extracellular molecule decay rate (used in dN/dt and dD/dt equations)
GI = 0.025   # NICD decay rate

# Initial and parameter perturbations
PI = 0.1
PP = 0.05

# Set the simulation length and initial cell state
TIME = 1000
STEPS = 10000
INITIAL_CELL = np.array([200, 200, 100])

# Define increasing hill function based on parameters above
@numba.njit
def H_PLUS(i):
    return (NM * i**K) / ((N0**K) + (i**K))


# Define decreasing hill function based on parameters below
@numba.njit
def H_MINUS(i):
    return (DM * D0**K) / ((D0**K) + (i**K))


