# Import libraries
import numpy as np
import sys
import logging
import time
import numba

# Import modules 
import visualization as vis
import domains
import gillespie 
import ode

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename = "main.log", filemode="w", level=logging.INFO)


# Run all simulations
np.random.seed(69420)
def main():

    # Two Cell Domain 
    domain = domains.linear(2, "dirichlet")

    # Deterministic Model (Linear ICs)
    linear_data = ode.ode(domain, "Deterministic-Linear", samples = 10)
    vis.two_cell_individual(domain, linear_data, "Linear Deterministic")
    
    # Deterministic Model (Random ICs)
    random_data = ode.ode(domain, "Deterministic-Random", samples = 10)
    vis.two_cell_individual(domain, random_data, "Random Deterministic")

    # Stochastic Model
    stochastic_data = ode.ode(domain, "Stochastic", samples = 10)
    vis.two_cell_individual(domain, stochastic_data, "Stochastic")

    # Gillespie Model
    gillespie_data = gillespie.gillespie(domain, samples = 10)
    vis.two_cell_individual(domain, gillespie_data, "Gillespie")

    # Comparison Plot
    all_data = (random_data, stochastic_data, gillespie_data)
    vis.two_cell_comparison(all_data)

    # Differentiation time plots (with additional samples)
    linear_data = ode.ode(domain, "Deterministic-Linear", samples = 50)
    stochastic_data = ode.ode(domain, "Stochastic", samples = 500)
    gillespie_data = gillespie.gillespie(domain, samples = 500)
    vis.two_cell_deterministic_differentiation(domain, linear_data)
    vis.two_cell_stochastic_differentiation(domain, stochastic_data, gillespie_data)
    
    # Linear Domain (Dirichlet)
    # domain, size = domains.linear(11, "dirichlet")
    # v_time, v_state = gillespie.run(domain, size)

    # Linear Domain (Periodic)
    # domain, size = domains.linear(11, "periodic")
    # v_time, v_state = gillespie.run(domain, size)


main()

