import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm

# --- Initialize Parameter Values ---
k = 2           # Hill exponent for Notch activation
NM = 10         # Maximum rates of Notch production
DM = 10         # Maximum rates of Delta production
N0 = 100        # Delta Hill function
D0 = 100        # Delta Hill function
KT = 0.0001     # Binding rate between extracellular Delta and Notch
G = 0.01        # Decay rate for Delta and Notch
GI = 0.025      # Decay rate for NICD

# --- Hill Functions ---
def H_plus(I):
    """Notch activation by intracellular NICD."""
    return (NM * I**k) / ((N0**k) + (I**k))

def H_minus(I):
    """Delta inhibited by intracellular NICD."""
    return (DM * D0**k) / ((D0**k) + (I**k))

# --- ODE System ---
def ode_system(t, y):
    N1, D1, I1, N2, D2, I2 = y
    
    # Cell 1
    dN1_dt = H_plus(I1) - KT * N1 * D2 - G * N1
    dD1_dt = H_minus(I1) - KT * D1 * N2 - G * D1
    dI1_dt = KT * N1 * D2 - GI * I1

    # Cell 2
    dN2_dt = H_plus(I2) - KT * N2 * D1 - G * N2
    dD2_dt = H_minus(I2) - KT * D2 * N1 - G * D2
    dI2_dt = KT * N2 * D1 - GI * I2

    return [dN1_dt, dD1_dt, dI1_dt, dN2_dt, dD2_dt, dI2_dt]

# --- Simulation Settings ---
t_span = (0, 1000)  
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Generate multiple initial conditions
num_simulations = 20
np.random.seed(42)  # For reproducibility

# Ranges for initial values
N_range = (100, 400)
D_range = (100, 400)
I_range = (50, 200)

# Store all solutions
all_solutions = []

# Run simulations with different initial conditions
for i in range(num_simulations):
    # Generate random initial conditions
    N = np.random.uniform(*N_range, 2)
    D = np.random.uniform(*D_range, 2)
    I = np.random.uniform(*I_range, 2)
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
    
    y0 = C1 + C2  # Concatenate the two cell initial conditions
    
    # Solve the ODEs
    sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    all_solutions.append(sol)

# --- Plotting all trajectories with mean ---
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
titles = ['Notch Level', 'Delta Level', 'NICD Level']
var_names = ['N', 'D', 'I']  # Variable names without the cell number

# Storage for calculating means
all_data = {
    'N1': np.zeros((num_simulations, len(t_eval))),
    'D1': np.zeros((num_simulations, len(t_eval))),
    'I1': np.zeros((num_simulations, len(t_eval))),
    'N2': np.zeros((num_simulations, len(t_eval))),
    'D2': np.zeros((num_simulations, len(t_eval))),
    'I2': np.zeros((num_simulations, len(t_eval)))
}

# Populate the arrays
for i, sol in enumerate(all_solutions):
    all_data['N1'][i] = sol.y[0]
    all_data['D1'][i] = sol.y[1]
    all_data['I1'][i] = sol.y[2]
    all_data['N2'][i] = sol.y[3]
    all_data['D2'][i] = sol.y[4]
    all_data['I2'][i] = sol.y[5]

# Calculate means
mean_data = {
    'N1': np.mean(all_data['N1'], axis=0),
    'D1': np.mean(all_data['D1'], axis=0),
    'I1': np.mean(all_data['I1'], axis=0),
    'N2': np.mean(all_data['N2'], axis=0),
    'D2': np.mean(all_data['D2'], axis=0),
    'I2': np.mean(all_data['I2'], axis=0)
}

# Plot each panel - using indices directly to avoid the KeyError
for idx, (title, var_name) in enumerate(zip(titles, var_names)):
    # Indices for cell 1 and cell 2 variables
    idx1 = idx
    idx2 = idx + 3
    
    # Plot individual trajectories with low opacity
    for sol in all_solutions:
        axs[idx].plot(sol.t, sol.y[idx1], 'b-', alpha=0.2)
        axs[idx].plot(sol.t, sol.y[idx2], 'r-', alpha=0.2)
    
    # Plot mean trajectories with high opacity
    cell1_key = f"{var_name}1"
    cell2_key = f"{var_name}2"
    
    axs[idx].plot(t_eval, mean_data[cell1_key], 'b-', linewidth=2, label=f'Mean {cell1_key}')
    axs[idx].plot(t_eval, mean_data[cell2_key], 'r-', linewidth=2, label=f'Mean {cell2_key}')
    
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel(title)
    axs[idx].legend(loc='best')
    
    # Add a grid for better readability
    axs[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

'''
# --- Phase portrait (optional) ---
plt.figure(figsize=(12, 6))

# Delta vs NICD for both cells
plt.subplot(1, 2, 1)
for sol in all_solutions:
    plt.plot(sol.y[2], sol.y[1], 'b-', alpha=0.2)  # I1 vs D1
plt.plot(mean_data['I1'], mean_data['D1'], 'b-', linewidth=2, label='Mean Cell 1')
plt.xlabel('NICD Level')
plt.ylabel('Delta Level')
plt.title('Phase Portrait: Delta vs NICD')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for sol in all_solutions:
    plt.plot(sol.y[5], sol.y[4], 'r-', alpha=0.5)  # I2 vs D2
plt.plot(mean_data['I2'], mean_data['D2'], 'r-', linewidth=2, label='Mean Cell 2')
plt.xlabel('NICD Level')
plt.ylabel('Delta Level')
plt.title('Phase Portrait: Delta vs NICD')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()'
'''