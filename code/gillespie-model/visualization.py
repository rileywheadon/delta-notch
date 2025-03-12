from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

# Create a static visualization of Delta levels over time
def linear(v_time, v_state):
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
