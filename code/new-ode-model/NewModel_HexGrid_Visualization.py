import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import RegularPolygon
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# --- Updated Parameter Values ---
k = 2           # Hill exponent for Notch activation
NM = 10         # Maximum rates of Notch production (increased from 10)
DM = 10         # Maximum rates of Delta production
N0 = 100        # Delta Hill function
D0 = 100        # Delta Hill function
KT = 0.0001     # Binding rate between estracellular Delta and Notch
G = 0.019        # Decay rate for Delta and Notch (increased from 0.01)
GI = 0.025      # Decay rate for NICD

# Hexagonal grid size parameters
nx = 8  # Number of cells in x direction
ny = 8  # Number of cells in y direction

# Animation settings
save_animation = False

# --- Hill Functions ---
def H_plus(I):
    """Notch activation by intracellular NICD."""
    return (NM * I**k) / ((N0**k) + (I**k))

def H_minus(I):
    """Delta inhibited by intracellular NICD."""
    return (DM * D0**k) / ((D0**k) + (I**k))

# --- Hexagonal Grid Helper Functions ---
def get_hex_neighbors(i, j, nx, ny):
    """
    Get the indices of the six neighboring hexagons with periodic boundary conditions.
    In a hexagonal grid, each cell has 6 neighbors.
    """
    # The neighbor pattern depends on whether we're in an even or odd row
    is_even_row = (i % 2 == 0)
    
    if is_even_row:
        # Even rows: neighbors are at positions:
        # (i-1,j-1), (i-1,j), (i,j-1), (i,j+1), (i+1,j-1), (i+1,j)
        neighbors = [
            ((i-1) % ny, (j-1) % nx),  # Top-left
            ((i-1) % ny, j),           # Top-right
            (i, (j-1) % nx),           # Left
            (i, (j+1) % nx),           # Right
            ((i+1) % ny, (j-1) % nx),  # Bottom-left
            ((i+1) % ny, j)            # Bottom-right
        ]
    else:
        # Odd rows: neighbors are at positions:
        # (i-1,j), (i-1,j+1), (i,j-1), (i,j+1), (i+1,j), (i+1,j+1)
        neighbors = [
            ((i-1) % ny, j),           # Top-left
            ((i-1) % ny, (j+1) % nx),  # Top-right
            (i, (j-1) % nx),           # Left
            (i, (j+1) % nx),           # Right
            ((i+1) % ny, j),           # Bottom-left
            ((i+1) % ny, (j+1) % nx)   # Bottom-right
        ]
    
    return neighbors

# --- ODE System ---
def ode_system(t, y):
    # Reshape into a 3D array where each cell has [N, D, I]
    state = y.reshape(ny, nx, 3)
    
    # Initialize the derivatives
    derivatives = np.zeros_like(state)
    
    for i in range(ny):
        for j in range(nx):
            # Get hexagonal neighbor indices with periodic boundary conditions
            neighbors = get_hex_neighbors(i, j, nx, ny)
            
            # Current cell values
            N_ij, D_ij, I_ij = state[i, j]
            
            # Calculate average Delta from neighbors
            D_neighbors_sum = 0
            for ni, nj in neighbors:
                D_neighbors_sum += state[ni, nj, 1]  # Delta value of neighbor
            D_neighbors = D_neighbors_sum / len(neighbors)
            
            # Calculate derivatives for current cell
            dN_dt = H_plus(I_ij) - KT * N_ij * D_neighbors - G * N_ij
            dD_dt = H_minus(I_ij) - G * D_ij
            dI_dt = KT * N_ij * D_neighbors - GI * I_ij
            
            derivatives[i, j] = [dN_dt, dD_dt, dI_dt]
    
    # Return flattened array
    return derivatives.flatten()

# --- Simulation Settings ---
t_span = (0, 5000)
t_eval = np.linspace(t_span[0], t_span[1], 200)  # Reduced number of time points for efficiency

# Initial conditions with small random perturbations for each cell
np.random.seed(42)  # For reproducibility
y0 = np.zeros(nx * ny * 3)
idx = 0
for i in range(ny):
    for j in range(nx):
        # Base values with some randomness
        N_init = 200 + np.random.normal(0, 20)
        D_init = 200 + np.random.normal(0, 20)
        I_init = 100 + np.random.normal(0, 10)
        
        # Ensure no negative values
        N_init = max(N_init, 1)
        D_init = max(D_init, 1)
        I_init = max(I_init, 1)
        
        # Set initial conditions for this cell
        y0[idx:idx+3] = [N_init, D_init, I_init]
        idx += 3

# --- Solve the ODEs ---
print(f"Solving the system of ODEs for {nx}x{ny} hexagonal grid ({nx*ny} cells)...")
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
print("ODE solution complete!")

# --- Extract Delta levels for visualization ---
time_points = sol.t
delta_data = np.zeros((ny, nx, len(time_points)))

# Extract Delta values for each cell over time
for t in range(len(time_points)):
    state_at_t = sol.y[:, t].reshape(ny, nx, 3)
    for i in range(ny):
        for j in range(nx):
            delta_data[i, j, t] = state_at_t[i, j, 1]  # Delta is at index 1

# Create a custom colormap for Delta levels (blue to red)
delta_cmap = LinearSegmentedColormap.from_list("DeltaMap", ["blue", "white", "red"])

# Calculate global min and max for consistent coloring
vmin = np.min(delta_data)
vmax = np.max(delta_data)
print(f"Delta value range: {vmin:.2f} to {vmax:.2f}")

# --- Create animation of the hexagonal grid over time ---
# Set up the figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

# Hexagon dimensions
radius = 1.0/2
hex_width = radius * 2
hex_height = np.sqrt(3) * radius

# Compute grid dimensions
width = nx * (1.5 * radius) + 0.5 * radius + 1
height = ny * hex_height + 0.5 * hex_height - 1.5

# Set axis limits with some padding
ax.set_xlim(-0.5, width + 0.5)
ax.set_ylim(-0.5, height + 0.5)
ax.set_title(f"Hexagonal Grid ({nx}×{ny}): Delta Expression Pattern Over Time")
ax.set_xticks([])
ax.set_yticks([])

# Create dictionary to hold hexagon patches and text objects
hexagons = {}
hex_texts = {}

# Initialize hexagons
for i in range(ny):
    for j in range(nx):
        # Calculate hexagon center position
        # For even rows, shift to align with odd rows
        if i % 2 == 0:
            # Even row
            x = j * 3.52/2 * radius
        else:
            # Odd row
            x = j * 3.52/2 * radius + 0.88 * radius
        
        y = i * hex_height / 1.15
        
        # Create hexagon patch
        hexagon = RegularPolygon(
            (x, y), 
            numVertices=6, 
            radius=radius,
            orientation=0,  # Flat-topped hexagon
            facecolor='white',
            edgecolor='black',
            linewidth=2.0
        )
        ax.add_patch(hexagon)
        hexagons[(i, j)] = hexagon
        
        # Add text for Delta value
        hex_texts[(i, j)] = ax.text(x, y, "", ha='center', va='center', fontsize=8)

# Add time text display
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12)

# Add colorbar
cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust position as needed
norm = plt.Normalize(vmin, vmax)
sm = plt.cm.ScalarMappable(cmap=delta_cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Delta Level')

# Animation update function
def update(frame):
    # Extract current Delta values at this time frame
    current_data = delta_data[:, :, frame]
    
    # Update all hexagons and texts
    objects_to_update = []
    
    for i in range(ny):
        for j in range(nx):
            # Get current delta value
            delta_val = current_data[i, j]
            
            # Normalize value for color mapping
            normalized_val = (delta_val - vmin) / (vmax - vmin)
            
            # Update hexagon color
            hexagons[(i, j)].set_facecolor(delta_cmap(normalized_val))
            objects_to_update.append(hexagons[(i, j)])
            
            # Update text
            hex_texts[(i, j)].set_text(f"{delta_val:.1f}")
            
            # Set text color based on background brightness
            text_color = 'white' if normalized_val > 0.6 or normalized_val < 0.3 else 'black'
            hex_texts[(i, j)].set_color(text_color)
            objects_to_update.append(hex_texts[(i, j)])
    
    # Update time text
    time_text.set_text(f'Time: {time_points[frame]:.1f}')
    objects_to_update.append(time_text)
    
    return objects_to_update

# Create animation (use every 4th frame for better performance)
frames = range(0, len(time_points), 4)
ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=100)

# Save animation
if save_animation == True:
    print("Saving animation...")
    ani.save('notch_delta_animation.gif', writer='pillow', fps=15)
    print("Animation saved!")

plt.tight_layout()
plt.show()

print("Animation complete!")