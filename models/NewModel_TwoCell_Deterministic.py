import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Initialize Parameter Values ---
k = 2           # Hill exponent for Notch activation
NM = 15         # Maximum rates of Notch production
DM = 10         # Maximum rates of Delta production
N0 = 100        # Delta Hill function
D0 = 100        # Delta Hill function
KT = 0.0001     # Binding rate between estracellular Delta and Notch
G = 0.02        # Decay rate for Delta and Notch
GI = 0.025       # Decay rate for NICD

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

# Initial conditions 
# C = [N, D, I]
C1 = [280, 200, 100]  # Initial conditions for cell 1
C2 = [300, 180, 120]  # Initial conditions for cell 2

# y = [N1, D1, I1, N2, D2, I2]
y0 = C1 + C2  # Concatenate the two cell initial conditions

# --- Solve the ODEs ---
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
'''
# --- Create the four-panel plot matching Collier paper ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Line width for the plots
lw = 1.0 

# Top left: n₁
axs[0, 0].plot(sol.t, sol.y[0], 'k-', linewidth=lw)
#axs[0, 0].set_xlim(0, 25)
#axs[0, 0].set_ylim(0, 1)
axs[0, 0].set_ylabel(r'$n_1$')
axs[0, 0].set_xlabel(r'$t$')

# Top right: d₁
axs[0, 1].plot(sol.t, sol.y[1], 'k-', linewidth=lw)
#axs[0, 1].set_xlim(0, 25)
#axs[0, 1].set_ylim(0, 1)
axs[0, 1].set_ylabel(r'$d_1$')
axs[0, 1].set_xlabel(r'$t$')

# Bottom left: n₂
axs[1, 0].plot(sol.t, sol.y[2], 'k-', linewidth=lw)
#axs[1, 0].set_xlim(0, 25)
#axs[1, 0].set_ylim(0, 1)
axs[1, 0].set_ylabel(r'$n_2$')
axs[1, 0].set_xlabel(r'$t$')

# Bottom right: d₂
axs[1, 1].plot(sol.t, sol.y[3], 'k-', linewidth=lw)
#axs[1, 1].set_xlim(0, 25)
#axs[1, 1].set_ylim(0, 1)
axs[1, 1].set_ylabel(r'$d_2$')
axs[1, 1].set_xlabel(r'$t$')

plt.tight_layout()
plt.show()
'''

# --- Plotting the Deterministic Dynamics ---
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(sol.t, sol.y[0], label='N1')
plt.plot(sol.t, sol.y[3], label='N2')
plt.xlabel('Time')
plt.ylabel('Notch Level')
plt.legend(loc='upper right')

plt.subplot(3, 1, 2)
plt.plot(sol.t, sol.y[1], label='D1')
plt.plot(sol.t, sol.y[4], label='D2')
plt.xlabel('Time')
plt.ylabel('Delta Level')
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.plot(sol.t, sol.y[2], label='NICD1')
plt.plot(sol.t, sol.y[5], label='NICD2')
plt.xlabel('Time')
plt.ylabel('NICD Level')
plt.legend()
plt.tight_layout()
plt.show()
