import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors

# Deterministic Delta-Notch system for hexagonal grid
def delta_notch_hex_grid(t, y, alpha_D, beta_D, alpha_N, beta_N, k, n, grid_size):
    """
    y is arranged as [D1, N1, D2, N2, ..., Dn, Nn]
    Each cell interacts with its six neighbors in a hexagonal grid
    with periodic boundary conditions (toroidal surface)
    """
    rows, cols = grid_size
    total_cells = rows * cols
    dydt = np.zeros_like(y)
    
    # Helper function to get the 1D index from 2D coordinates
    def get_index(r, c):
        r %= rows  # Wrap around rows
        c %= cols  # Wrap around columns
        return 2 * (r * cols + c)
    
    # For each cell, calculate its derivatives
    for r in range(rows):
        for c in range(cols):
            cell_idx = r * cols + c
            D_idx = 2 * cell_idx
            N_idx = D_idx + 1
            
            # Define neighbor offsets for hexagonal grid
            if r % 2 == 0:  # Even row
                neighbor_offsets = [
                    (-1, -1), (-1, 0),   # Top-left, Top-right
                    (0, -1), (0, 1),     # Left, Right
                    (1, -1), (1, 0)      # Bottom-left, Bottom-right
                ]
            else:  # Odd row
                neighbor_offsets = [
                    (-1, 0), (-1, 1),    # Top-left, Top-right
                    (0, -1), (0, 1),     # Left, Right
                    (1, 0), (1, 1)       # Bottom-left, Bottom-right
                ]
            
            # Get neighbor Delta values
            neighbor_deltas = []
            for dr, dc in neighbor_offsets:
                nr, nc = r + dr, c + dc
                neighbor_idx = get_index(nr, nc)
                neighbor_deltas.append(y[neighbor_idx])
            
            # Calculate Delta inhibition by cell's own Notch
            D_inhibition = 1 / (1 + (y[N_idx] / k) ** n)
            
            # Calculate Notch activation by average of neighboring Delta
            neighbor_delta_avg = sum(neighbor_deltas) / len(neighbor_deltas)
            N_activation = (neighbor_delta_avg ** n) / (k ** n + neighbor_delta_avg ** n)
            
            dydt[D_idx] = alpha_D * D_inhibition - beta_D * y[D_idx]
            dydt[N_idx] = alpha_N * N_activation - beta_N * y[N_idx]
    
    return dydt

# Stochastic version of the Delta-Notch system
def stochastic_delta_notch_hex_grid(y, dt, alpha_D, beta_D, alpha_N, beta_N, 
                                   k, n, grid_size, sigma_D, sigma_N):
    """
    Stochastic version with additive noise
    Returns updated state after one time step
    """
    # Calculate deterministic component
    dydt = delta_notch_hex_grid(0, y, alpha_D, beta_D, alpha_N, beta_N, k, n, grid_size)
    
    # Add stochastic component
    noise = np.zeros_like(y)
    noise[::2] = sigma_D * np.sqrt(dt) * np.random.randn(len(y) // 2)  # Delta noise
    noise[1::2] = sigma_N * np.sqrt(dt) * np.random.randn(len(y) // 2) # Notch noise
    
    # Update with Euler-Maruyama step
    y_new = y + dt * np.array(dydt) + noise
    
    # Ensure non-negative concentrations
    y_new = np.clip(y_new, 0, None)
    
    return y_new

# Parameters with added noise terms
params = {
    'alpha_D': 1.0, 'beta_D': 0.1,
    'alpha_N': 1.0, 'beta_N': 0.1,
    'k': 0.5, 'n': 3,
    'sigma_D': 0.08, 'sigma_N': 0.05  # Noise intensities
}

# Simulation parameters
grid_size = (10, 12)
rows, cols = grid_size
total_cells = rows * cols
dt = 0.5  # Reduced time step for stability
t_span = (0, 150)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
n_steps = len(t_eval)

# Initialize system with larger random variations
y = np.zeros((2 * total_cells, n_steps))  # 2D array to store all time steps
y[:, 0] = [0.5 + 0.3 * np.random.random() for _ in range(2 * total_cells)]

# Stochastic simulation loop
for i in range(1, n_steps):
    y[:, i] = stochastic_delta_notch_hex_grid(
        y[:, i - 1], dt, **params, grid_size=grid_size
    )

# Visualization setup
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_aspect('equal')
ax.axis('off')

# Hexagon placement and initialization
hex_size = 1.0
hex_height = np.sqrt(3) * hex_size
y_offset = 1.0

def get_hex_coordinates(row, col):
    x = col * (3.51 / 2) * hex_size
    y = row * hex_height / 1.15 + y_offset
    if row % 2 == 1:
        x += 0.89 * hex_size
    return x, y

hexagons = []
all_x = []
all_y = []
for r in range(rows):
    for c in range(cols):
        x_coord, y_coord = get_hex_coordinates(r, c)
        all_x.append(x_coord)
        all_y.append(y_coord)
        hex_cell = RegularPolygon(
            (x_coord, y_coord), 
            numVertices=6, 
            radius=hex_size,  
            orientation=0,    # Flat-topped hexagon
            edgecolor='k',    # Thin black border
            linewidth=1.0,
            animated=True
        )
        ax.add_patch(hex_cell)
        hexagons.append(hex_cell)

# Compute boundaries based on actual hexagon positions and a small margin
margin = 2.5
xmin, xmax = min(all_x) - margin, max(all_x) + margin
ymin, ymax = min(all_y) - margin, max(all_y) + margin

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Create a time text annotation near the top
time_text = ax.text((xmin + xmax) / 2, ymax - margin / 3,
                    'Time: {:.1f}'.format(0),
                    ha='center', fontsize=14, animated=True)

# Extract Delta values for normalization
print(y)
D_values = y[::2, :]  # Extract all Delta values
vmin, vmax = np.quantile(D_values, [0.02, 0.98])  # Robust scaling

cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
cbar.set_label('Delta Concentration', fontsize=10)

def update(frame):
    """Update cell colors with dynamic scaling"""
    updates = []
    current_D = y[::2, frame]  # All Delta values
    
    # Update colors
    for i, hex_cell in enumerate(hexagons):
        color = cmap(norm(current_D[i]))
        hex_cell.set_facecolor(color)
        updates.append(hex_cell)
    
    # Update time display
    time_text.set_text(f'Time: {t_eval[frame]:.1f}')
    updates.append(time_text)
    
    return updates

# Create animation with faster refresh rate
ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_steps,
    interval=30,  # Faster animation
    blit=True
)

plt.title('Stochastic Delta-Notch Patterning in Hexagonal Epithelium', fontsize=14)
plt.tight_layout()
plt.show()
