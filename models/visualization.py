# Import libraries
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Import modules
from config import TIME, STEPS


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


def vis13(params, success):
    return
