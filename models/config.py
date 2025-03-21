import numpy as np
import numba

# Default simulation parameters
DEFAULT = np.array([
    2,       # K: Hill function exponent
    10,      # NM: Maximum rate of Notch production
    10,      # DM: Maximum rate of Delta production
    100,     # N0: Notch hill function (H+) half-max
    100,     # D0: Delta hill function (H-) half-max
    0.0001,  # KT: Delta-Notch binding rate
    0.02,    # G: Extracellular molecule decay rate 
    0.025,   # GI: NICD decay rate
])

# Default initial cell state
DEFAULT_CELL = np.array([200, 200, 100])

# Default simulation length and time steps
TIME = 1500
STEPS = 7500

# Define increasing hill function based on parameters above
@numba.njit
def H_PLUS(params, i):
    K, NM, DM, N0, D0 = params[:5]
    return (NM * i**K) / ((N0**K) + (i**K))

# Define decreasing hill function based on parameters below
@numba.njit
def H_MINUS(params, i):
    K, NM, DM, N0, D0 = params[:5]
    return (DM * D0**K) / ((D0**K) + (i**K))


