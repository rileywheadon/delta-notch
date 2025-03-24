import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# --- Initialize Parameter Values ---
k = 2           # Hill exponent for Notch activation
NM = 10         # Maximum rates of Notch production
DM = 10         # Maximum rates of Delta production
N0 = 100        # Delta Hill function
D0 = 100        # Delta Hill function
KT = 0.0001     # Binding rate between estracellular Delta and Notch
G = 0.02        # Decay rate for Delta and Notch
GI = 0.025      # Decay rate for NICD

# Number of cells in the row
num_cells = 8

# --- Hill Functions ---
def H_plus(I):
    """Notch activation by intracellular NICD."""
    return (NM * I**k) / ((N0**k) + (I**k))

def H_minus(I):
    """Delta inhibited by intracellular NICD."""
    return (DM * D0**k) / ((D0**k) + (I**k))

# --- ODE System ---
def ode_system(t, y):
    # Reshape into a matrix where each row represents a cell [N, D, I]
    state = y.reshape(num_cells, 3)
    
    # Initialize the derivatives
    derivatives = np.zeros_like(state)
    
    for i in range(num_cells):
        # Get left and right neighbor indices with periodic boundary conditions
        left_idx = (i - 1) % num_cells
        right_idx = (i + 1) % num_cells
        
        # Current cell values
        N_i, D_i, I_i = state[i]
        
        # Neighbor Delta values (for Notch-Delta binding)
        D_left = state[left_idx][1]
        D_right = state[right_idx][1]
        
        # Average Delta signal from neighbors
        D_neighbors = (D_left + D_right) / 2
        
        # Calculate derivatives for current cell
        dN_dt = H_plus(I_i) - KT * N_i * D_neighbors - G * N_i
        dD_dt = H_minus(I_i) - G * D_i
        dI_dt = KT * N_i * D_neighbors - GI * I_i
        
        derivatives[i] = [dN_dt, dD_dt, dI_dt]
    
    # Return flattened array
    return derivatives.flatten()

# --- Simulation Settings ---
t_span = (0, 1000)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Initial conditions with small random perturbations for each cell
np.random.seed(42)  # For reproducibility
y0 = np.zeros(num_cells * 3)

for i in range(num_cells):
    # Base values with some randomness
    N_init = 200 + np.random.normal(0, 20)
    D_init = 200 + np.random.normal(0, 20)
    I_init = 100 + np.random.normal(0, 10)
    
    # Ensure no negative values
    N_init = max(N_init, 1)
    D_init = max(D_init, 1)
    I_init = max(I_init, 1)
    
    # Set initial conditions for this cell
    y0[i*3:(i+1)*3] = [N_init, D_init, I_init]

# --- Solve the ODEs ---
print("Solving the system of ODEs...")
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
print("ODE solution complete!")

# --- Visualization ---
# Extract results for plotting
time_points = sol.t
results = sol.y.reshape(num_cells * 3, -1)

# Create a custom colormap for Delta levels (blue to red)
delta_cmap = LinearSegmentedColormap.from_list("DeltaMap", ["blue", "white", "red"])

# --- Static visualization of Delta patterns over time ---
plt.figure(figsize=(12, 8))

# Subplot 1: Line plot of Delta levels for each cell
plt.subplot(3, 1, 1)
for i in range(num_cells):
    delta_idx = i*3 + 1  # Index for Delta in the flattened array
    plt.plot(time_points, results[delta_idx], label=f'Cell {i+1}')
plt.xlabel('Time')
plt.ylabel('Delta Level')
plt.title('Delta Expression Over Time for Each Cell')
plt.legend(loc='upper right', ncol=2)

# Subplot 2: Heatmap of Delta expression (cells vs time)
plt.subplot(3, 1, 2)
delta_data = np.zeros((num_cells, len(time_points)))
for i in range(num_cells):
    delta_idx = i*3 + 1
    delta_data[i] = results[delta_idx]

# Normalize for better visualization
vmin = np.min(delta_data)
vmax = np.max(delta_data)
plt.imshow(delta_data, aspect='auto', cmap=delta_cmap, 
           extent=[0, time_points[-1], -0.5, num_cells-0.5], vmin=vmin, vmax=vmax)
plt.colorbar(label='Delta Level')
plt.ylabel('Cell Number')
plt.xlabel('Time')
plt.title('Heatmap of Delta Expression Patterns')
plt.yticks(range(num_cells))

# Subplot 3: Cell row visualization at the final time point
ax = plt.subplot(3, 1, 3)
final_delta = delta_data[:, -1]
normalized_delta = (final_delta - vmin) / (vmax - vmin)  # normalize to [0, 1]

# Draw cells as rectangles colored by Delta level
for i in range(num_cells):
    rect = plt.Rectangle((i, 0), 1, 1, facecolor=delta_cmap(normalized_delta[i]), edgecolor='black')
    ax.add_patch(rect)
    plt.text(i + 0.5, 0.5, f"{final_delta[i]:.1f}", ha='center', va='center')

plt.xlim(0, num_cells)
plt.ylim(0, 1)
plt.title('Final Delta Expression Pattern')
plt.xticks(np.arange(0.5, num_cells, 1), [f'Cell {i+1}' for i in range(num_cells)])
plt.yticks([])

plt.tight_layout()
plt.show()  # Display the static plots

# --- Create interactive animation of cells over time ---
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_xlim(0, num_cells)
ax.set_ylim(0, 1)
ax.set_title('Delta Expression Pattern Over Time')
ax.set_xticks(np.arange(0.5, num_cells, 1))
ax.set_xticklabels([f'Cell {i+1}' for i in range(num_cells)])
ax.set_yticks([])

# Create initial rectangles for cells
rects = []
texts = []
for i in range(num_cells):
    rect = plt.Rectangle((i, 0), 1, 1, facecolor='white', edgecolor='black')
    ax.add_patch(rect)
    rects.append(rect)
    text = ax.text(i + 0.5, 0.5, "", ha='center', va='center')
    texts.append(text)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animation update function
def update(frame):
    current_delta = delta_data[:, frame]
    normalized_delta = (current_delta - vmin) / (vmax - vmin)
    
    for i in range(num_cells):
        rects[i].set_facecolor(delta_cmap(normalized_delta[i]))
        texts[i].set_text(f"{current_delta[i]:.1f}")
    
    time_text.set_text(f'Time: {time_points[frame]:.1f}')
    return rects + texts + [time_text]

# Create animation (use every 5th frame to speed up animation)
frames = range(0, len(time_points), 5)
ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)

plt.tight_layout()
plt.show()
