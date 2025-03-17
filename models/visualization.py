# Import libraries
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Import modules
from config import TIME, STEPS


# Static, individual visualization for two-cell models
def two_cell_individual(domain, data, model):

    name, neighbours, size = domain
    times, states, means, stds, diffs = data
    fig, axs = plt.subplots(nrows = 3, sharex = True)

    # Iterate through axes (0: Notch, 1: Delta, 2: NICD)
    for i in range(3):
        
        # Reduce the linewidth
        mpl.rcParams['lines.linewidth'] = 1

        # Plot the trajectories for each sample
        for j in range(len(states)):
            axs[i].plot(times[j], states[j][:, 0, i], color = "green", alpha = 0.2)
            axs[i].plot(times[j], states[j][:, 1, i], color = "orange", alpha = 0.2)

        # Plot the means at full opacity
        vT = np.linspace(0, TIME, STEPS)
        axs[i].plot(vT, means[:, 0, i], color = "darkgreen")
        axs[i].plot(vT, means[:, 1, i], color = "darkorange")

        # Plot the standard deviations
        axs[i].plot(vT, means[:, 0, i] + stds[:, 0, i], "--", color = "green")
        axs[i].plot(vT, means[:, 0, i] - stds[:, 0, i], "--", color = "green")
        axs[i].plot(vT, means[:, 1, i] + stds[:, 1, i], "--", color = "orange")
        axs[i].plot(vT, means[:, 1, i] - stds[:, 1, i], "--", color = "orange")

    # Set axis labels
    axs[0].set_ylabel("Notch Molecules")
    axs[1].set_ylabel("Delta Molecules")
    axs[2].set_ylabel("NICD Molecules")

    # Add title and labels to the graph
    fig.supxlabel("Time")
    fig.suptitle(f"{name} {model}")

    # Save to the img/ foler
    plt.tight_layout()
    fname = "_".join(f"{name} {model}".lower().split(" "))
    plt.savefig(f"img/{fname}.pdf", dpi = 200)
    plt.close(fig)


# Static, comparison visualization for two-cell models
def two_cell_comparison(all_data):

    # Iterate through axes (0: Deterministic, 1: Stochastic, 2: Gillespie)
    fig, axs = plt.subplots(nrows = 3, sharex = True)
    for i in range(3):

        # Unpack the data arrays
        data = all_data[i]
        times, states, means, stds, diffs = data

        # Plot the trajectories for each sample
        for j in range(len(states)):
            axs[i].plot(times[j], states[j][:, 0, 0], color = "green", alpha = 0.1)
            axs[i].plot(times[j], states[j][:, 1, 0], color = "orange", alpha = 0.1)

        # Plot the means at full opacity
        default_times = np.linspace(0, TIME, STEPS)
        axs[i].plot(default_times, means[:, 0, 0], color = "green")
        axs[i].plot(default_times, means[:, 1, 0], color = "orange")

    # Set axis labels
    axs[0].set_ylabel("Deterministic")
    axs[1].set_ylabel("Stochastic")
    axs[2].set_ylabel("Gillespie")

    # Add title and labels to the graph
    fig.supxlabel("Time")
    fig.supylabel("Notch Molecules")
    fig.suptitle(f"Two Cell Model Comparison")

    # Save to the img/ foler
    plt.tight_layout()
    plt.savefig(f"img/two_cell_comparison.pdf", dpi = 200)
    plt.close(fig)


# Static, differentiation time plot for deterministic two-cell models
def two_cell_deterministic_differentiation(domain, data):

    # Unpack domain and data objects
    name, neighbours, size = domain
    states, diffs = np.array(data[1]), data[4]

    # Compute the initial perturbations
    perturbations = states[:, 0, 0, 0] - states[:, 0, 1, 0]

    # Plot the initial perturbation against differentiation time
    mpl.rcParams['lines.linewidth'] = 2
    plt.plot(perturbations, diffs, "ko-")
    plt.xlabel("Initial Perturbation") 
    plt.ylabel("Differentiation Time")
    plt.title("Initial Perturbation vs. Differentiation Time")

    # Save to the img/ folder
    plt.tight_layout()
    plt.savefig(f"img/two_cell_deterministic_differentiation.pdf", dpi = 200)
    plt.close()


# Static differentiation time plot for stochastic ODE and gillespie models
def two_cell_stochastic_differentiation(domain, stochastic_data, gillespie_data):

    # Unpack domain and data objects (abbreviate s - stochastic, g - gillespie)
    name, neighbours, size = domain
    s_diffs = stochastic_data[4]
    g_diffs = gillespie_data[4]

    # Plot the initial perturbation against differentiation time
    bins = np.linspace(0, 1500, 30)
    plt.hist(s_diffs, bins, color="blue", label = "Stochastic", alpha = 0.4)
    plt.hist(g_diffs, bins, color="red", label = "Gillespie", alpha = 0.4)
    plt.xlabel("Differentiation Time") 
    plt.ylabel("Count")
    plt.title("Histogram of Differentiation Times")
    plt.legend()

    # Save to the img/ folder
    plt.tight_layout()
    plt.savefig(f"img/two_cell_stochastic_differentiation.pdf", dpi = 200)
    plt.close()


# Static, individual visualization for linear models
def linear_individual(v_time, v_state):
    cells = v_state.shape[1]

    # Create a custom colormap for Delta levels (blue to red)
    delta_cmap = LinearSegmentedColormap.from_list("DeltaMap", ["blue", "white", "red"])

    # --- Static visualization of Delta patterns over time ---
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)

    # Subplot 1: Line plot of Delta levels for each cell
    for i in range(cells):
        plt.plot(v_time, v_state[:, i, 1], label=f'Cell {i+1}')

    plt.xlabel('Time')
    plt.ylabel('Delta Level')
    plt.title('Delta Expression Over Time for Each Cell')
    plt.legend(loc='upper right', ncol=2)

    # Subplot 3: Cell row visualization at the final time point
    ax = plt.subplot(3, 1, 3)
    d_final = v_state[-1, :, 1]
    d_normal = (d_final - np.min(d_final)) / (np.max(d_final) - np.min(d_final)) 

    # Draw cells as rectangles colored by Delta level
    for i in range(cells):
        color = delta_cmap(d_normal[i])
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black')
        plt.text(i + 0.5, 0.5, f"{d_final[i]:.1f}", ha='center', va='center')
        ax.add_patch(rect)

    plt.xlim(0, cells)
    plt.ylim(0, 1)
    plt.title('Final Delta Expression Pattern')
    plt.xticks(np.arange(0.5, cells, 1), [f'Cell {i+1}' for i in range(cells)])
    plt.yticks([])

    # Display the static plots
    plt.tight_layout()
    plt.show()  
