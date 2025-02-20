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
# Choose initial conditions that break symmetry
y0 = [0.1, 0.9, 0.15, 0.85]

# --- Solve the ODEs ---
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval)

# --- Plotting the Deterministic Dynamics ---
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], label='N1')
plt.plot(sol.t, sol.y[2], label='N2')
plt.xlabel('Time')
plt.ylabel('Notch Level')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[1], label='D1')
plt.plot(sol.t, sol.y[3], label='D2')
plt.xlabel('Time')
plt.ylabel('Delta Level')
plt.legend()
plt.tight_layout()
plt.show()
