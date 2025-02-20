import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors

# Delta-Notch system for hexagonal grid (unchanged from corrected version)
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

# Parameters
params = {
    'alpha_D': 1.0, 'beta_D': 0.1,
    'alpha_N': 1.0, 'beta_N': 0.1,
    'k': 0.5, 'n': 3
}

# Grid dimensions
grid_size = (10, 12)  # (rows, cols)
rows, cols = grid_size
total_cells = rows * cols

# Initial conditions with random variations
y0 = []
for i in range(total_cells):
    D_init = 0.5 + 0.1 * np.random.random()
    N_init = 0.5 + 0.1 * np.random.random()
    y0.extend([D_init, N_init])

# Time span for simulation
t_span = (0, 150)
t_eval = np.linspace(*t_span, 300)

# Solve the system
sol = solve_ivp(
    delta_notch_hex_grid,
    t_span,
    y0,
    args=(*params.values(), grid_size),
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-9
)

# Visualization setup with a smaller figure size
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_aspect('equal')
ax.axis('off')

# Each hexagon has side length 1
hex_size = 1.0
hex_height = np.sqrt(3) * hex_size

# Shift the entire grid upward by adding a y-offset.
y_offset = 1.0

def get_hex_coordinates(row, col):
    # For proper tiling, odd rows are offset in x.
    x = col * (3.51 / 2) * hex_size
    y = row * hex_height / 1.15 + y_offset  # shifted upward by y_offset
    if row % 2 == 1:
        x += 0.89 * hex_size
    return x, y

# Create hexagonal cells and collect coordinates for setting limits.
hexagons = []
all_x = []
all_y = []
for r in range(rows):
    for c in range(cols):
        x, y = get_hex_coordinates(r, c)
        all_x.append(x)
        all_y.append(y)
        hex_cell = RegularPolygon(
            (x, y), 
            numVertices=6, 
            radius=hex_size,  
            orientation=0,    # Flat-topped hexagon
            edgecolor='k',    # Thin black border
            linewidth=1.0,
            animated=True
        )
        ax.add_patch(hex_cell)
        hexagons.append(hex_cell)

# Compute boundaries based on actual hexagon positions and a small margin.
margin = 2.5
xmin, xmax = min(all_x) - margin, max(all_x) + margin
ymin, ymax = min(all_y) - margin, max(all_y) + margin

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Create a time text annotation near the top (but inside the window)
time_text = ax.text((xmin + xmax) / 2, ymax - margin/3,
                    'Time: {:.1f}'.format(0),
                    ha='center', fontsize=14, animated=True)

# Extract Delta values for normalization
D_values = np.zeros((total_cells, len(t_eval)))
for i in range(total_cells):
    D_values[i] = sol.y[2*i]
min_D = np.min(D_values)
max_D = np.max(D_values)

# Create a colormap and add a colorbar
cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=min_D, vmax=max_D)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
cbar.set_label('Delta Concentration', fontsize=10)

def update(frame):
    """Update cell colors and time text based on current time point."""
    updates = []
    for i in range(total_cells):
        current_D = sol.y[2*i][frame]
        color = cmap(norm(current_D))
        hexagons[i].set_facecolor(color)
        updates.append(hexagons[i])
    time_text.set_text('Time: {:.1f}'.format(t_eval[frame]))
    updates.append(time_text)
    return updates

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(t_eval),
    interval=50,
    blit=True
)

plt.title('Delta-Notch Patterning in Hexagonal Epithelium', fontsize=14)
plt.tight_layout()
plt.show()
