import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def run_stochastic_delta_notch(noise_levels=None, num_simulations=1, seed=None, dt=0.1, t_max=1000, 
                               initial_conditions=None):
    """
    Run multiple Delta-Notch interaction simulations with stochastic differential equations.
    
    Parameters:
    -----------
    noise_levels : dict or None
        Dictionary with keys 'N', 'D', 'I' and values representing the
        noise intensity (sigma) for each variable. If None, no noise is added.
    num_simulations : int
        Number of simulations to run with different noise realizations.
    seed : int or None
        Seed for random number generation for reproducibility.
    dt : float
        Time step for Euler-Maruyama integration.
    t_max : float
        Maximum simulation time.
    initial_conditions : dict or None
        Dictionary with keys 'N1', 'D1', 'I1', 'N2', 'D2', 'I2' specifying
        initial conditions. If None, random initial conditions are generated.
        
    Returns:
    --------
    all_solutions : list
        List of dictionaries containing simulation results
    params : dict
        Dictionary of parameter values used in the simulations
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # --- Initialize Default Parameter Values ---
    params = {
        'k': 2,          # Hill exponent for Notch activation
        'NM': 10,        # Maximum rates of Notch production
        'DM': 10,        # Maximum rates of Delta production
        'N0': 100,       # Delta Hill function
        'D0': 100,       # Delta Hill function
        'KT': 0.0001,    # Binding rate between extracellular Delta and Notch
        'G': 0.02,       # Decay rate for Delta and Notch
        'GI': 0.025      # Decay rate for NICD
    }
    
    # Default noise levels if none provided
    if noise_levels is None:
        noise_levels = {'N': 0.0, 'D': 0.0, 'I': 0.0}
    
    # Ensure all noise components are defined
    for key in ['N', 'D', 'I']:
        if key not in noise_levels:
            noise_levels[key] = 0.0
    
    # --- Define Hill Functions ---
    def H_plus(I):
        """Notch activation by intracellular NICD."""
        return (params['NM'] * I**params['k']) / ((params['N0']**params['k']) + (I**params['k']))

    def H_minus(I):
        """Delta inhibited by intracellular NICD."""
        return (params['DM'] * params['D0']**params['k']) / ((params['D0']**params['k']) + (I**params['k']))

    # --- Deterministic ODE System ---
    def deterministic_step(N1, D1, I1, N2, D2, I2):
        # Cell 1
        dN1_dt = H_plus(I1) - params['KT'] * N1 * D2 - params['G'] * N1
        dD1_dt = H_minus(I1) - params['KT'] * D1 * N2 - params['G'] * D1
        dI1_dt = params['KT'] * N1 * D2 - params['GI'] * I1

        # Cell 2
        dN2_dt = H_plus(I2) - params['KT'] * N2 * D1 - params['G'] * N2
        dD2_dt = H_minus(I2) - params['KT'] * D2 * N1 - params['G'] * D2
        dI2_dt = params['KT'] * N2 * D1 - params['GI'] * I2

        return dN1_dt, dD1_dt, dI1_dt, dN2_dt, dD2_dt, dI2_dt

    # --- Simulation Settings ---
    t_points = np.arange(0, t_max + dt, dt)
    num_steps = len(t_points)
    
    # --- Initialize starting conditions ---
    if initial_conditions is None:
        # Ranges for initial values
        N_range = (100, 400)
        D_range = (100, 400)
        I_range = (50, 200)
        
        # Generate a single random initial condition for all simulations
        N = np.random.uniform(*N_range, 2)
        D = np.random.uniform(*D_range, 2)
        I = np.random.uniform(*I_range, 2)
        
        # Initial configurations
        C1 = [
            max(N),  # N1
            min(D),  # D1
            max(I)   # I1
        ]
        
        C2 = [
            min(N),  # N2
            max(D),  # D2
            min(I)   # I2
        ]
        
        y0 = C1 + C2  # [N1, D1, I1, N2, D2, I2]
    else:
        # Use user-provided initial conditions
        y0 = [
            initial_conditions.get('N1', 300),  # Default values if not specified
            initial_conditions.get('D1', 100),
            initial_conditions.get('I1', 150),
            initial_conditions.get('N2', 100),
            initial_conditions.get('D2', 300),
            initial_conditions.get('I2', 50)
        ]
    
    # Store all solutions
    all_solutions = []
    
    # Run multiple simulations with same initial condition but different noise realizations
    for sim in range(num_simulations):
        # Initialize solution arrays
        N1_sol = np.zeros(num_steps)
        D1_sol = np.zeros(num_steps)
        I1_sol = np.zeros(num_steps)
        N2_sol = np.zeros(num_steps)
        D2_sol = np.zeros(num_steps)
        I2_sol = np.zeros(num_steps)
        
        # Set initial conditions
        N1_sol[0] = y0[0]
        D1_sol[0] = y0[1]
        I1_sol[0] = y0[2]
        N2_sol[0] = y0[3]
        D2_sol[0] = y0[4]
        I2_sol[0] = y0[5]
        
        # Euler-Maruyama integration
        for i in range(1, num_steps):
            # Current state
            N1, D1, I1 = N1_sol[i-1], D1_sol[i-1], I1_sol[i-1]
            N2, D2, I2 = N2_sol[i-1], D2_sol[i-1], I2_sol[i-1]
            
            # Deterministic step
            dN1, dD1, dI1, dN2, dD2, dI2 = deterministic_step(N1, D1, I1, N2, D2, I2)
            
            # Generate Wiener increments (scaled by sqrt(dt) for proper Brownian motion)
            dW_N1 = np.random.normal(0, np.sqrt(dt))
            dW_D1 = np.random.normal(0, np.sqrt(dt))
            dW_I1 = np.random.normal(0, np.sqrt(dt))
            dW_N2 = np.random.normal(0, np.sqrt(dt))
            dW_D2 = np.random.normal(0, np.sqrt(dt))
            dW_I2 = np.random.normal(0, np.sqrt(dt))
            
            # Apply stochastic update with noise scaling
            N1_sol[i] = max(0, N1 + dN1 * dt + noise_levels['N'] * N1 * dW_N1)
            D1_sol[i] = max(0, D1 + dD1 * dt + noise_levels['D'] * D1 * dW_D1)
            I1_sol[i] = max(0, I1 + dI1 * dt + noise_levels['I'] * I1 * dW_I1)
            N2_sol[i] = max(0, N2 + dN2 * dt + noise_levels['N'] * N2 * dW_N2)
            D2_sol[i] = max(0, D2 + dD2 * dt + noise_levels['D'] * D2 * dW_D2)
            I2_sol[i] = max(0, I2 + dI2 * dt + noise_levels['I'] * I2 * dW_I2)
        
        # Store solution
        solution = {
            't': t_points,
            'y': [N1_sol, D1_sol, I1_sol, N2_sol, D2_sol, I2_sol]
        }
        all_solutions.append(solution)
    
    return all_solutions, params

def plot_fate_based_simulations(all_solutions, params, noise_levels):
    """
    Plot the results of multiple stochastic simulations, showing trajectories 
    based on cell fate (primary vs secondary) determined by sorted final Notch levels.
    
    Parameters:
    -----------
    all_solutions : list
        List of dictionaries containing simulation results
    params : dict
        Dictionary of parameter values
    noise_levels : dict
        Dictionary with noise levels for each variable
    """
    num_simulations = len(all_solutions)
    
    # Get time points from the first solution
    t_eval = all_solutions[0]['t']
    
    # --- Create arrays to store all cell trajectories ---
    all_cells = {
        'N': [],  # Notch trajectories
        'D': [],  # Delta trajectories
        'I': [],  # NICD trajectories
        'final_N': []  # Final Notch values for sorting
    }
    
    # Extract data from all simulations and all cells
    for sol in all_solutions:
        # Cell 1
        all_cells['N'].append(sol['y'][0])  # N1
        all_cells['D'].append(sol['y'][1])  # D1
        all_cells['I'].append(sol['y'][2])  # I1
        all_cells['final_N'].append(sol['y'][0][-1])  # Final N1 value
        
        # Cell 2
        all_cells['N'].append(sol['y'][3])  # N2
        all_cells['D'].append(sol['y'][4])  # D2
        all_cells['I'].append(sol['y'][5])  # I2
        all_cells['final_N'].append(sol['y'][3][-1])  # Final N2 value
    
    # Convert lists to arrays for easier handling
    for key in ['N', 'D', 'I']:
        all_cells[key] = np.array(all_cells[key])
    
    # Get sorting indices based on final Notch values
    sorted_indices = np.argsort(all_cells['final_N'])
    
    # Total number of cells
    total_cells = len(sorted_indices)
    
    # Determine half point for primary vs secondary fate
    # Primary fate: lower half of Notch values
    # Secondary fate: upper half of Notch values
    mid_point = total_cells // 2
    
    # Split into primary and secondary fates
    primary_indices = sorted_indices[:mid_point]
    secondary_indices = sorted_indices[mid_point:]
    
    # --- Plotting all trajectories with mean ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    titles = ['Notch Level', 'Delta Level', 'NICD Level']
    var_names = ['N', 'D', 'I']  # Variable names
    
    # Plot each panel
    for idx, (title, var_name) in enumerate(zip(titles, var_names)):
        # Plot individual trajectories with low opacity
        # Primary fate (lower Notch)
        for i in primary_indices:
            axs[idx].plot(t_eval, all_cells[var_name][i], 'b-', alpha=0.1)
        
        # Secondary fate (higher Notch)
        for i in secondary_indices:
            axs[idx].plot(t_eval, all_cells[var_name][i], 'r-', alpha=0.1)
        
        # Calculate means for each fate
        mean_primary = np.mean(all_cells[var_name][primary_indices], axis=0)
        mean_secondary = np.mean(all_cells[var_name][secondary_indices], axis=0)
        
        # Calculate standard deviations for each fate
        std_primary = np.std(all_cells[var_name][primary_indices], axis=0)
        std_secondary = np.std(all_cells[var_name][secondary_indices], axis=0)
        
        # Add shaded regions for standard deviation
        axs[idx].fill_between(t_eval, 
                             mean_primary - std_primary,
                             mean_primary + std_primary,
                             color='blue', alpha=0.0)
        
        axs[idx].fill_between(t_eval, 
                             mean_secondary - std_secondary,
                             mean_secondary + std_secondary,
                             color='red', alpha=0.0)
        
        # Plot mean lines
        axs[idx].plot(t_eval, mean_primary, 'b-', linewidth=1.5, 
                     label=f'Primary Fate (Low Notch) n={len(primary_indices)}')
        
        axs[idx].plot(t_eval, mean_secondary, 'r-', linewidth=1.5, 
                     label=f'Secondary Fate (High Notch) n={len(secondary_indices)}')
        
        axs[idx].set_xlabel('Time')
        axs[idx].set_ylabel(title)
        #axs[idx].legend(loc='best')
        
        # Add a grid for better readability
        axs[idx].grid(True, alpha=0.3)
    
    # Add a text box with parameter and noise information
    param_text = f"Number of Simulations: {num_simulations}\n\n"
    
    # Add noise level information
    param_text += "Noise Levels (σ):\n"
    for name, value in noise_levels.items():
        param_text += f"{name}: {value:.4f}\n"
    
    param_text += "\nModel Parameters:\n"
    for name, value in params.items():
        param_text += f"{name}: {value:.4f}\n"
    
    # Place the text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.82, 0.5, param_text, transform=fig.transFigure, fontsize=9,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout(rect=[0, 0, 0.75, 0.95])  # Make room for parameter text
    plt.suptitle('Delta-Notch Signaling: Primary vs Secondary Cell Fate', fontsize=14)
    plt.show()
    return fig

# Example usage demonstration
if __name__ == "__main__":
    # Example 1: Custom initial conditions with minimal noise
    custom_initial_conditions = {
        'N1': 200, 'D1': 200, 'I1': 100,
        'N2': 200, 'D2': 200, 'I2': 100
    }
    
    noise_levels1 = {
        'N': 0.02,    # 2% noise for Notch
        'D': 0.02,    # 2% noise for Delta
        'I': 0.02     # 2% noise for NICD
    }
    
    all_solutions1, params1 = run_stochastic_delta_notch(
        noise_levels=noise_levels1,
        num_simulations=25,
        seed=42,
        initial_conditions=custom_initial_conditions
    )
    plot_fate_based_simulations(all_solutions1, params1, noise_levels1)
    '''
    # Example 2: Random initial conditions with differential noise levels
    noise_levels2 = {
        'N': 0.05,    # 5% noise for Notch
        'D': 0.05,    # 5% noise for Delta
        'I': 0.03     # 3% noise for NICD
    }
    
    all_solutions2, params2 = run_stochastic_delta_notch(
        noise_levels=noise_levels2,
        num_simulations=10,
        seed=42,
        dt=0.1,       # Time step
        t_max=1000    # Maximum simulation time
    )
    plot_fate_based_simulations(all_solutions2, params2, noise_levels2)
    '''