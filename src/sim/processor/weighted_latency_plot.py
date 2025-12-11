import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401

from utils.plot_style import set_plot_style
from sim.simulator import weighted_time_by_priority


if __name__ == "__main__":
    set_plot_style()
    # plot weighted latency by priority and frequency
    # time = N / frequency, where N is the number of cycles and is constant
    # weighted_time = weighted_time_by_priority(time, priority)

    # create frequencies array to be 0.1 to 3.0 GHz with step size 0.1
    frequencies = np.arange(0.1, 2.1, 0.05)
    N = 1e-3  # number of cycles
    priorities = [0, 1, 2, 3, 4, 5]

    # plot 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(frequencies, priorities)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            time = N / X[i, j] * 1e3  # convert s to ns
            Z[i, j] = weighted_time_by_priority(time, Y[i, j])
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Priority')
    ax.set_zlabel('Weighted Latency (ns)')
    plt.tight_layout()
    plt.savefig('outputs/weighted_latency_by_priority_frequency.png', dpi=300)




    



 