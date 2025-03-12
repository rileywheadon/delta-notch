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

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename = "main.log", filemode="w", level=logging.INFO)


# Run all simulations
def main():

    # Two Cell Domain
    domain, size = domains.linear(2, "dirichlet")
    gillespie_data = gillespie.run(domain, size, samples = 20)
    visualization.two_cell_individual(gillespie_data, "Two-Cell Gillespie")
    

    # Linear Domain (Dirichlet)
    # domain, size = domains.linear(11, "dirichlet")
    # v_time, v_state = gillespie.run(domain, size)

    # Linear Domain (Periodic)
    # domain, size = domains.linear(11, "periodic")
    # v_time, v_state = gillespie.run(domain, size)


main()

