import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Revised Parameter Choices ---
k = 2.0       # Hill exponent for Notch activation
h = 2.0       # Hill exponent for Delta inhibition
beta = 1.5    # Production scaling for Notch; >1 to promote instability
theta = 0.5   # Threshold for Delta activation of Notch
mu = 0.5      # Threshold for Notch inhibition of Delta

# --- Revised Hill Functions ---
def f(D, k, theta):
    """
    Notch activation by Delta from the neighboring cell.
    A threshold parameter theta sharpens the response.
    """
    return D**k / (theta**k + D**k)

def g(N, h, mu):
    """
    Delta production inhibited by own Notch.
    The parameter mu sets the inhibition threshold.
    """
    return 1 / (1 + (N/mu)**h)

# --- Define the Revised ODE System ---
def ode_system(t, y):
    # y = [N1, D1, N2, D2] for the two-cell system.
    N1, D1, N2, D2 = y
    dN1_dt = beta * f(D2, k, theta) - N1
    dD1_dt = g(N1, h, mu) - D1
    dN2_dt = beta * f(D1, k, theta) - N2
    dD2_dt = g(N2, h, mu) - D2
    return [dN1_dt, dD1_dt, dN2_dt, dD2_dt]

# --- Simulation Settings ---
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Number of different initial conditions to simulate
num_simulations = 30

# Prepare an array to store each solution (shape: num_simulations x 4 x len(t_eval))
all_solutions = []

# Loop over simulations with different starting positions.
for i in range(num_simulations):
    # Create a random perturbation about the baseline initial condition.
    # Baseline: [0.1, 0.9, 0.15, 0.85]
    y0 = [
        0.92 + np.random.uniform(-0.05, 0.05),
        0.92 + np.random.uniform(-0.05, 0.05),
        0.92 + np.random.uniform(-0.05, 0.05),
        0.92 + np.random.uniform(-0.05, 0.05)
    ]
    sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-9)
    all_solutions.append(sol.y)

# Convert list to array: shape (num_simulations, 4, len(t_eval))
all_solutions = np.array(all_solutions)
# Compute the mean solution (averaging over simulations)
mean_solution = all_solutions.mean(axis=0)  # shape: (4, len(t_eval))

# --- Plotting ---
plt.figure(figsize=(10, 8))

# Colors for clarity:
color_N1 = 'tab:blue'
color_N2 = 'tab:orange'
color_D1 = 'tab:blue'
color_D2 = 'tab:orange'

# --- Plot Notch Dynamics (Top subplot) ---
plt.subplot(2, 1, 1)
# Plot individual simulations with low alpha (faint lines)
for sol in all_solutions:
    plt.plot(t_eval, sol[0], color=color_N1, alpha=0.3)
    plt.plot(t_eval, sol[2], color=color_N2, alpha=0.3)
# Plot the mean solutions prominently (thicker lines)
plt.plot(t_eval, mean_solution[0], color=color_N1, lw=3, label='Mean N1')
plt.plot(t_eval, mean_solution[2], color=color_N2, lw=3, label='Mean N2')
plt.xlabel('Time')
plt.ylabel('Notch Level')
plt.title('Notch Dynamics')
plt.legend()

# --- Plot Delta Dynamics (Bottom subplot) ---
plt.subplot(2, 1, 2)
# Plot individual simulations with low alpha
for sol in all_solutions:
    plt.plot(t_eval, sol[1], color=color_D1, alpha=0.3)
    plt.plot(t_eval, sol[3], color=color_D2, alpha=0.3)
# Plot the mean solutions prominently
plt.plot(t_eval, mean_solution[1], color=color_D1, lw=3, label='Mean D1')
plt.plot(t_eval, mean_solution[3], color=color_D2, lw=3, label='Mean D2')
plt.xlabel('Time')
plt.ylabel('Delta Level')
plt.title('Delta Dynamics')
plt.legend()

plt.tight_layout()
plt.show()
