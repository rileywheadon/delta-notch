import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Adjusted Parameter Choices ---
k = 2.0         # Hill exponent for Notch activation
h = 2.0         # Hill exponent for Delta inhibition
v = 1.0         # Production scaling for Delta
theta = 0.01    # Threshold for Delta activation of Notch ('a' in paper)
mu = 100        # Threshold for Notch inhibition of Delta ('b' in paper)

# --- Hill Functions ---
def f(D, k, theta):
    """Notch activation by Delta from neighboring cell."""
    return D**k / (theta + D**k)

def g(N, h, mu):
    """Delta production inhibited by own Notch."""
    return 1 / (1 + mu * (N)**h)

# --- ODE System ---
def ode_system(t, y):
    N1, D1, N2, D2 = y
    dN1_dt = f(D2/2, k, theta) - N1
    dD1_dt = v * (g(N1, h, mu) - D1)
    dN2_dt = f(D1/2, k, theta) - N2
    dD2_dt = v * (g(N2, h, mu) - D2)
    return [dN1_dt, dD1_dt, dN2_dt, dD2_dt]

# --- Adjusted Simulation Settings ---
t_span = (0, 25)  
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Initial conditions 
# y = [N1, D1, N2, D2]
y0 = [1.0, 1.0, 0.99, 0.99]  # Slight asymmetry in N2,D2 initial values

# --- Solve the ODEs ---
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

# --- Create the four-panel plot matching the image ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Line width for the plots
lw = 1.0 

# Top left: n₁
axs[0, 0].plot(sol.t, sol.y[0], 'k-', linewidth=lw)
axs[0, 0].set_xlim(0, 25)
axs[0, 0].set_ylim(0, 1)
axs[0, 0].set_ylabel(r'$n_1$')
axs[0, 0].set_xlabel(r'$t$')

# Top right: d₁
axs[0, 1].plot(sol.t, sol.y[1], 'k-', linewidth=lw)
axs[0, 1].set_xlim(0, 25)
axs[0, 1].set_ylim(0, 1)
axs[0, 1].set_ylabel(r'$d_1$')
axs[0, 1].set_xlabel(r'$t$')

# Bottom left: n₂
axs[1, 0].plot(sol.t, sol.y[2], 'k-', linewidth=lw)
axs[1, 0].set_xlim(0, 25)
axs[1, 0].set_ylim(0, 1)
axs[1, 0].set_ylabel(r'$n_2$')
axs[1, 0].set_xlabel(r'$t$')

# Bottom right: d₂
axs[1, 1].plot(sol.t, sol.y[3], 'k-', linewidth=lw)
axs[1, 1].set_xlim(0, 25)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].set_ylabel(r'$d_2$')
axs[1, 1].set_xlabel(r'$t$')

plt.tight_layout()
plt.show()
