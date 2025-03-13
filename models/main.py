# Import libraries
import numpy as np
import sys
import logging
import time
import numba

# Import modules 
import visualization
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

    # Deterministic Model
    deterministic_data = ode.ode(domain, "Deterministic", samples = 25)
    visualization.two_cell_individual(domain, deterministic_data, "Deterministic")

    # Stochastic Model
    stochastic_data = ode.ode(domain, "Stochastic", samples = 25)
    visualization.two_cell_individual(domain, stochastic_data, "Stochastic")

    # Gillespie Model
    gillespie_data = gillespie.gillespie(domain, samples = 25)
    visualization.two_cell_individual(domain, gillespie_data, "Gillespie")

    # Comparison Plot
    all_data = (deterministic_data, stochastic_data, gillespie_data)
    visualization.two_cell_comparison(all_data)
    
    # Linear Domain (Dirichlet)
    # domain, size = domains.linear(11, "dirichlet")
    # v_time, v_state = gillespie.run(domain, size)

    # Linear Domain (Periodic)
    # domain, size = domains.linear(11, "periodic")
    # v_time, v_state = gillespie.run(domain, size)


main()

