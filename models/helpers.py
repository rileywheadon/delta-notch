import numpy as np
from scipy.interpolate import interp1d


# Interpolate states vector at p evenly spaced points
def interpolate(points, vT, vS):
    times = np.linspace(vT.min(), vT.max(), points)
    return times, interp1d(vT, vS, axis = 0)(times)


# Get the mean and standard deviation over all simulations
def summarize(states):
    return states.mean(axis = 0), states.std(axis = 0)


# Get differentiation times for a single simulation
def differentiation(vT, vS):

    # Sort and subset the state vector by final Notch concentration
    vS = np.sort(vS[:, :, 0], axis = 1)
    vS = vS[:, vS[-1, :] < 1]

    # If no cells have final Notch less than 1, return the maximum differentiation time 
    if (vS.shape[1] == 0):
        return vT[-1]

    # Otherwise determine the maximum differentiation time from the remaining cells
    vI = np.argmax(vS[:, 0] < 1, axis = 0)
    return vT[np.max(vI)]



