# Import libraries
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, RegularPolygon, Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Import modules
from config import TIME, STEPS, DEFAULT


# Visualization 01-03 helper function:
def vis_01_03(times, states, means, stds, name, title):

    # Iterate through axes (0: Notch, 1: Delta, 2: NICD)
    fig, axs = plt.subplots(nrows = 3, sharex = True)
    for i in range(3):
        mpl.rcParams['lines.linewidth'] = 1

        # Plot the trajectories for each sample
        for j in range(len(states)):
            axs[i].plot(times, states[j, :, 0, i], color = "green", alpha = 0.2)
            axs[i].plot(times, states[j, :, 1, i], color = "orange", alpha = 0.2)

        # Plot the means at full opacity
        axs[i].plot(times, means[:, 0, i], color = "darkgreen")
        axs[i].plot(times, means[:, 1, i], color = "darkorange")

        # Plot the standard deviations
        axs[i].plot(times, means[:, 0, i] + stds[:, 0, i], "--", color = "green")
        axs[i].plot(times, means[:, 0, i] - stds[:, 0, i], "--", color = "green")
        axs[i].plot(times, means[:, 1, i] + stds[:, 1, i], "--", color = "orange")
        axs[i].plot(times, means[:, 1, i] - stds[:, 1, i], "--", color = "orange")

    # Set axis labels
    axs[0].set_ylabel("Notch Molecules")
    axs[1].set_ylabel("Delta Molecules")
    axs[2].set_ylabel("NICD Molecules")

    # Add title and labels to the graph
    fig.supxlabel("Time")
    fig.suptitle(title)

    # Save to the img/ foler
    plt.tight_layout()
    plt.savefig(f"img/{name}.pdf", dpi = 200)
    plt.close(fig)


def vis01(times, states, means, stds):
    vis_01_03(times, states, means, stds, "vis01", "Deterministic ODE Model")

def vis02(times, states, means, stds):
    vis_01_03(times, states, means, stds, "vis02", "Stochastic ODE Model")

def vis03(times, states, means, stds):
    vis_01_03(times, states, means, stds, "vis03", "Agent-Based Model")


# Visualization 04-06 helper function:
def vis_04_06(vT, vS, name, title):

    size = vS.shape[1]
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, layout = "constrained")
    cmap = LinearSegmentedColormap.from_list("NotchMap", ["blue", "white", "red"])

    # Subplot 1: Line plot of Notch levels for each cell
    for i in range(size):
        ax0.plot(vT, vS[:, i, 0], label=f'Cell {i+1}')

    ax0.set_xlabel('Time')
    ax0.set_ylabel('Notch Molecules')
    ax0.set_title('Notch Expression Over Time for Each Cell')
    plt.figlegend(fontsize = 10)

    # Subplot 2: Heatmap of Notch expression (cells vs time)
    heatmap = ax1.imshow(
        vS[:, :, 0].T, 
        aspect = 'auto',
        cmap = cmap,
        extent= [0, vT[-1], 0.5, size+0.5], 
        interpolation = 'none'
    )

    ax1.set_ylabel('Cell Number')
    ax1.set_yticks(np.arange(0, size, 2) + 1)
    ax1.set_xlabel('Time')
    ax1.set_title('Heatmap of Notch Expression Patterns')

    # Subplot 3: Notch visualization at the final time point
    max_notch = vS[:, :, 0].max()
    min_notch = vS[:, :, 0].min()
    notch = (vS[-1, :, 0] - min_notch) / (max_notch - min_notch)

    # Draw cells as rectangles colored by Delta level
    for i in range(size):
        rect = plt.Rectangle((i + 0.5, 0), 1, 1, facecolor=cmap(notch[i]))
        ax2.add_patch(rect)
        ax2.text(i + 1, 0.5, f"{int(vS[-1, i, 0])}", ha='center', va='center')

    ax2.set_xlim(0 + 0.5, size + 0.5)
    ax2.set_xticks(np.arange(0, size) + 1)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_title('Final Notch Expression Pattern')

    # Set the figure title and add a colorbar
    fig.suptitle(title)
    fig.colorbar(heatmap, ax = [ax1, ax2])

    # Display the plot
    plt.savefig(f"img/{name}.pdf", dpi = 200)
    plt.close()  


def vis04(vT, vS):
    vis_04_06(vT, vS, "vis04", "Deterministic ODE Model")

def vis05(vT, vS):
    vis_04_06(vT, vS, "vis05", "Stochastic ODE Model")

def vis06(vT, vS):
    vis_04_06(vT, vS, "vis06", "Agent-Based Model")


def vis10(shifts, dtimes):

    # Plot the initial perturbation against differentiation time
    mpl.rcParams['lines.linewidth'] = 2
    plt.plot(shifts, dtimes, "k-")
    plt.xlabel("Initial Perturbation (# of Notch Molecules)") 
    plt.ylabel("Differentiation Time")
    plt.title("Initial Perturbation vs. Differentiation Time")

    # Save to the img/ folder
    plt.tight_layout()
    plt.savefig(f"img/vis10.pdf", dpi = 200)
    plt.close()


def vis11(noise, dtimes):

    # Plot the histograms
    VT = np.linspace(0, TIME, STEPS)
    for i in range(len(noise)):
        kde = gaussian_kde(dtimes[i])
        label = "Agent-Based" if noise[i] == 0 else f"Noise: {noise[i]}"
        plt.plot(VT, kde(VT), label = label)

    plt.xlabel("Differentiation Time") 
    plt.ylabel("Density")
    plt.title("KDE of Differentiation Times")
    plt.legend()

    # Save to the img/ folder
    plt.tight_layout()
    plt.savefig(f"img/vis11.pdf", dpi = 200)
    plt.close()


def vis12(shift, fates):

    # Plot the initial shift against the fate proportion
    plt.plot(shift, fates, "ko-")
    plt.xlabel("Initial Perturbation (# of Notch Molecules)")
    plt.ylabel("Proportion of Primary Outcomes")
    plt.title("Effect of Initial Perturbation on Primary Outcome")
    
    # Save to the img/ folder
    plt.tight_layout()
    plt.savefig(f"img/vis12.pdf", dpi = 200)
    plt.close()


# Helper function for vis13
def stability_visualization(ax, grids, pair):
    
    # List of parameter names in LaTeX
    NAMES = [
        r"$K$",
        r"$N_M$",
        r"$D_M$",
        r"$N_0$",
        r"$D_0$",
        r"$K_T$",
        r"$\gamma$",
        r"$\gamma_I$"
    ]

    # Unpack parameters, set constants
    i1, i2 = pair
    d_grid, s_grid, g_grid = grids

    # Compute the convex hull for each grid
    d_hull = d_grid[ConvexHull(d_grid).vertices, :]
    s_hull = s_grid[ConvexHull(s_grid).vertices, :]
    g_hull = g_grid[ConvexHull(g_grid).vertices, :]

    # Plot the convex hull for each grid
    d_poly = Polygon(d_hull, fc='none', ec="b", label="Deterministic ODE")
    s_poly = Polygon(s_hull, fc='none', ec="g", label="Stochastic ODE")
    g_poly = Polygon(g_hull, fc='none', ec="r", label="Agent-Based")

    # Add the patches to the axes object
    ax.add_patch(d_poly)
    ax.add_patch(s_poly)
    ax.add_patch(g_poly)
   
    # Set tick formatters
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))

    # Set the x and y limits, titles, labels, etc.
    ax.xaxis.set_ticks(np.linspace(0, DEFAULT[i1] * 10, 3))
    ax.yaxis.set_ticks(np.linspace(0, DEFAULT[i2] * 10, 3))
    ax.set_xlim(0, DEFAULT[i1] * 10)
    ax.set_ylim(0, DEFAULT[i2] * 10)
    ax.set_xlabel(NAMES[i1])
    ax.set_ylabel(NAMES[i2])
    ax.set_title(f"{NAMES[i1]} vs. {NAMES[i2]}")

    # Remove the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return [d_poly, s_poly, g_poly]


# Produce a triangle plot for stability analysis
def vis13(grids, pairs): 

    # The final plot will have the following arrangement:
    # x/y  0      1      2      3
    # 0 (1, 2)
    # 1 (1, 5) (2, 5)
    # 2 (1, 6) (2, 6) (5, 6)
    # 3 (1, 7) (2, 7) (5, 7) (6, 7)

    plt.rcParams['font.size'] = 10
    fig, axs = plt.subplots(4, 4, figsize = (15, 12))
    plots = [(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3)]
    pairs = [(1,2),(1,5),(2,5),(1,6),(2,6),(5,6),(1,7),(2,7),(5,7),(6,7)]

    # Iterate through the subplots, drawing when necessary
    handles = 0
    for (x, y), grid, pair in zip(plots, grids, pairs):
        handles = stability_visualization(axs[x][y], grid, pair)

    # Delete the empty plots
    for (x, y) in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        axs[x][y].clear()
        axs[x][y].axis("off")

    # Add a legend 
    plt.figlegend(handles = handles)
    fig.suptitle("Stability Analysis")

    # Save to the img/ folder
    plt.tight_layout()
    plt.savefig(f"img/vis13.pdf", dpi = 200)
    plt.close()

#  Produces a visualization of the cell pattern for either hexagonal or linear domains
def visualize_pattern(pattern, domain='hexagonal', ny=7, nx=7, radius=1.0):
    """
    Visualize a pattern in either hexagonal or linear mode:
    
    Hexagonal Mode:
    - pattern is a 1D numpy array of 0s and 1s
    - ny is the number of rows
    - nx is the number of columns
    - radius controls the size of hexagons
    
    Linear Mode:
    - pattern is a 1D numpy array of 0s and 1s
    - Visualizes a single row of rectangular cells
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if domain == 'hexagonal':
        # Ensure the pattern matches the grid size
        assert len(pattern) == ny * nx, "Pattern length must match grid size"
        
        # Hexagon dimensions
        hex_height = np.sqrt(3) * radius
        
        # Initialize hexagons in a parallelogram grid
        pattern_index = 0
        for j in range(ny):
            for l in range(nx):
                # Calculate hexagon center position for parallelogram grid
                # Shift each row to right while starting from top
                x = l * np.sqrt(3) * radius + j * 0.866 * radius  # Shifted right
                y = (ny - 1 - j) * hex_height * 0.866  # Invert y to start from top
                
                # Determine color based on the grid value
                facecolor = 'black' if pattern[pattern_index] == 0 else 'white'
                pattern_index += 1
                
                # Create hexagon patch
                hexagon = RegularPolygon(
                    (x, y), 
                    numVertices=6, 
                    radius=radius,
                    orientation=0,  # Flat-topped hexagon
                    facecolor=facecolor,
                    edgecolor='gray',
                    linewidth=2.0
                )
                ax.add_patch(hexagon)
        
        # Set axis limits based on the parallelogram grid dimensions
        max_x = (nx - 1) * 1.5 * radius + (ny - 1) * 0.75 * radius + 2 * radius 
        max_y = (ny - 1) * hex_height * 0.866 + 2 * radius
        ax.set_xlim(-radius, max_x + radius)
        ax.set_ylim(-radius, max_y + radius)
        
        # Title
        plt.title(f'Hexagonal Grid Pattern ({ny}x{nx})')
    
    elif domain == 'linear':
        # Ensure the pattern is a 1D array
        assert len(pattern.shape) == 1, "Pattern must be a 1D array for linear mode"
        
        # Rectangle width and height
        rect_width = 1.0
        rect_height = 2.0
        
        # Create rectangles for each cell in the pattern
        for i, cell_value in enumerate(pattern):
            # Determine color based on the cell value
            facecolor = 'black' if cell_value == 0 else 'white'
            
            # Create rectangle patch
            rect = Rectangle(
                (i * rect_width, 0),  # x, y position
                rect_width,  # width
                rect_height,  # height
                facecolor=facecolor,
                edgecolor='gray',
                linewidth=2.0
            )
            ax.add_patch(rect)
        
        # Set axis limits
        ax.set_xlim(-0.1, len(pattern) * rect_width + 0.1)
        ax.set_ylim(-0.1, rect_height + 0.1)
        
        # Title
        plt.title(f'Linear Pattern ({len(pattern)} cells)')
    
    else:
        raise ValueError("Mode must be either 'hexagonal' or 'linear'")
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    # Remove axis
    ax.axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

    # Example usages
    # Hexagonal domain
    # hex_pattern = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 
    #            1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 
    #            1, 1, 0, 1, 1, 0, 0, 1, 0])
    # visualize_pattern(hex_pattern, domain='hexagonal', ny=7, nx=7)

    # Linear domain
    # linear_pattern = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    # visualize_pattern(linear_pattern, domain='linear')

