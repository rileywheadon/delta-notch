# Import libraries
import numpy as np
import time
import numba
from tqdm import trange

# Import modules 
from config import DEFAULT_CELL, DEFAULT
from helpers import *
import visualization as vis
import domains
import gillespie
import ode

# Experiment 01: Deterministic ODE model
def ex01(runs = 10, pb = 0.05):

    # Run 10 simulations of the deterministic model
    print("\nRunning Experiment 01\n")
    times = np.empty((100))
    states = np.empty((runs, 100, 2, 3))
    domain = domains.linear(2, "dirichlet") 
    for i in trange(runs):

        # Generate initial conditions
        perturbations = np.random.normal(0, DEFAULT_CELL * pb, (2, 3))
        initial = DEFAULT_CELL + perturbations

        # Run the simulation, sort the cells, interpolate the results
        vT, vS = ode.ode(domain, initial)
        vS = vS[:, vS[-1, :, 0].argsort(), :]
        times, states[i] = interpolate(100, vT, vS)
    
    # Compute summary statistics, plot the results
    means, stds = summarize(states)
    vis.vis01(times, states, means, stds)
    print("\nExperiment Completed!\n")


# Experiment 02: Stochastic ODE model
def ex02(runs = 10, noise = 0.02):

    # Run 10 simulations of the deterministic model
    print("\nRunning Experiment 02\n")
    times = np.empty((100))
    states = np.empty((runs, 100, 2, 3))
    domain = domains.linear(2, "dirichlet") 
    for i in trange(runs):

        # Run the simulation, sort the cells, interpolate the results
        vT, vS = ode.ode(domain, DEFAULT_CELL, noise = noise)
        vS = vS[:, vS[-1, :, 0].argsort(), :]
        times, states[i] = interpolate(100, vT, vS)
    
    # Compute summary statistics, plot the results
    means, stds = summarize(states)
    vis.vis02(times, states, means, stds)
    print("\nExperiment Completed!\n")


# Experiment 03: Agent-Based Model
def ex03(runs = 10):

    # Run 10 simulations of the deterministic model
    print("\nRunning Experiment 03\n")
    times = np.empty((100))
    states = np.empty((runs, 100, 2, 3))
    domain = domains.linear(2, "dirichlet") 
    for i in trange(runs):

        # Run the simulation, sort the cells, interpolate the results
        initial = np.tile(DEFAULT_CELL, (2, 1))
        vT, vS = gillespie.gillespie(domain, initial)
        vT, vS = np.array(vT), np.array(vS)
        vS = vS[:, vS[-1, :, 0].argsort(), :]
        times, states[i] = interpolate(100, vT, vS)
    
    # Compute summary statistics, plot the results
    means, stds = summarize(states)
    vis.vis03(times, states, means, stds)
    print("\nExperiment Completed!\n")


# Experiment 04: Deterministic ODE model, linear domain
def ex04(size = 7, pb = 0.05):

    # Run 1 simulations of the deterministic model
    print("\nRunning Experiment 04\n")
    domain = domains.linear(size, "dirichlet") 

    # Generate initial conditions
    perturbations = np.random.normal(0, DEFAULT_CELL * pb, (size, 3))
    initial = DEFAULT_CELL + perturbations

    # Run the simulation, plot the results
    vT, vS = ode.ode(domain, initial)
    vT, vS = interpolate(100, vT, vS)
    vis.vis04(vT, vS)
    print("\nExperiment Completed!\n")


# Experiment 05: Stochastic ODE model, linear domain
def ex05(size = 7, noise = 0.02):

    # Run 1 simulation of the stochastic model
    print("\nRunning Experiment 05\n")
    domain = domains.linear(size, "dirichlet") 
    vT, vS = ode.ode(domain, DEFAULT_CELL, noise = noise)
    vT, vS = interpolate(100, vT, vS)
    vis.vis05(vT, vS)
    print("\nExperiment Completed!\n")


# Experiment 06: Agent-based model, linear domain
def ex06(size = 7):

    # Run 1 simulation of the agent-based model
    print("\nRunning Experiment 06\n")
    domain = domains.linear(size, "dirichlet") 
    initial = np.tile(DEFAULT_CELL, (size, 1))
    vT, vS = gillespie.gillespie(domain, initial)
    vT, vS = np.array(vT), np.array(vS)
    vT, vS = interpolate(100, vT, vS)
    vis.vis06(vT, vS)
    print("\nExperiment Completed!\n")


# Experiment 10: Deterministic two-cell differentiation times
def ex10(runs = 500, shift = 0.2):

    # Run simulations of the deterministic model
    print("\nRunning Experiment 10\n")
    shifts = np.empty((runs))
    dtimes = np.empty((runs))
    domain = domains.linear(2, "dirichlet") 
    for i in trange(runs):

        # Set the initial conditions
        SHIFT = np.array([(i + 1) * (shift / 2), 0, 0])
        initial = np.array([DEFAULT_CELL + SHIFT, DEFAULT_CELL - SHIFT])

        # Run the simulation, sort the cells, find the differentiation time
        vT, vS = ode.ode(domain, initial, time = 2000, steps = 10000)
        shifts[i] = i * shift
        dtimes[i] = differentiation(vT, vS)
    
    # Plot the results
    vis.vis10(shifts, dtimes)
    print("\nExperiment Completed!\n")


# Experiment 11: Effect of noise on stochastic differentiation times
def ex11(runs = 500, noise = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]):

    # Run simulations of the agent-based and deterministic models
    print("\nRunning Experiment 11\n")
    noises = np.repeat(noise, runs)
    dtimes = np.empty((runs * len(noise)))
    domain = domains.linear(2, "dirichlet") 
    for i in trange(runs * len(noise)):

        # If noise is 0, run the Gillespie model
        if noises[i] == 0: 
            initial = np.tile(DEFAULT_CELL, (2, 1))
            vT, vS = gillespie.gillespie(domain, initial)
            vT, vS = np.array(vT), np.array(vS)

        # Otherwise run the stochastic ODE model
        else:
            vT, vS = ode.ode(domain, DEFAULT_CELL, noise = noises[i])

        # Get the differentiation time
        dtimes[i] = differentiation(vT, vS)

    # Reshape dtimes
    dtimes = np.reshape(dtimes, (len(noise), runs))
    
    # Plot the results
    vis.vis11(noise, dtimes)
    print("\nExperiment Completed!\n")


# Experiment 12: Effect of initial perturbation on cell fates in agent-based model
def ex12(runs = 100, shift = np.linspace(0, 100, 6)):

    # Run simulations of the agent-based model
    print("\nRunning Experiment 12\n")
    shifts = np.repeat(shift, runs)
    fates = np.empty(len(shifts))
    domain = domains.linear(2, "dirichlet") 
    for i in trange(len(shifts)):

        # Set the initial condition
        S = np.array([shifts[i] / 2, 0, 0])
        initial = np.array([DEFAULT_CELL + S, DEFAULT_CELL - S])

        # Run the simulation
        vT, vS = gillespie.gillespie(domain, initial)
        vT, vS = np.array(vT), np.array(vS)

        # Append 0 to fates if the first cell had 0 Notch, 1 otherwise
        fates[i] = 0 if vS[-1, 0, 0] < 1 else 1

    # Compute the mean of the cell fates
    means = np.reshape(fates, (len(shift), runs)).mean(axis = 1)

    # Plot the results
    vis.vis12(shift, means)
    print("\nExperiment Completed!\n")


# Helper function for stability analysis on two parameters i1, i2 in DEFAULT
def two_parameter_stability(i1, i2, model, samples = 9):

    # Helper function to generate parameter space for parameter i
    def parameter_space(i):
        if i in [i1, i2]:
            return DEFAULT[i] * np.logspace(-1, 1, samples)
        else:
            return [DEFAULT[i]]
  
    # Generate list of parameter arrays for each simulation (plist)
    grid = np.meshgrid(*[parameter_space(i) for i in range(8)])
    params = np.array(grid).T.reshape(-1,8)
    success = np.empty(len(params))

    # Iterate through the list of parameters
    domain = domains.linear(2, "Dirichlet")
    for i in trange(len(params)):

        # Set the initial condition (adding 2% initial perturbation for deterministic)
        initial = np.array([DEFAULT_CELL, DEFAULT_CELL], dtype = np.float64)
        if model == "Deterministic":
            initial += np.random.normal(0, DEFAULT_CELL * 0.02, (2, 3))

        # Run a simulation with the given model (with 2% noise for stochastic ODE)
        vT, vS = None, None
        if model == "Deterministic":
            vT, vS = ode.ode(domain, initial, params = params[i])
        if model == "Stochastic":
            vT, vS = ode.ode(domain, initial, noise = 0.02, params = params[i])
        if model == "Gillespie":
            vT, vS = gillespie.gillespie(domain, initial, params = params[i])

        # Check if one cell has low Notch and the other has high Notch
        vS = np.array(vS)
        vS = np.sort(vS[:, :, 0], axis = 1)
        success[i] = (vS[-1, 0] < 1 and vS[-1, 1] > 1)

    # Subset the parameter list on the successful simulations
    return params[np.argwhere(success), [i1, i2]]


# Experiment 13: Effect of parameters on cell differentiation in all three models 
def ex13():

    print("\nRunning Experiment 13\n")

    # Define the paris of indices for stability analysis 
    # Use the parameters NM (1), DM (2), KT (5), G (6), GI (7)
    pairs = [(1,2),(1,5),(2,5),(1,6),(2,6),(5,6),(1,7),(2,7),(5,7),(6,7)]
    grids = []
    for pair in pairs:
        d_grid = two_parameter_stability(pair[0], pair[1], "Deterministic")
        s_grid = two_parameter_stability(pair[0], pair[1], "Stochastic")
        g_grid = two_parameter_stability(pair[0], pair[1], "Gillespie")
        grids.append([d_grid, s_grid, g_grid])

    vis.vis13(grids, pairs)
    print("\nExperiment Completed!\n")


