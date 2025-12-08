import matplotlib.pyplot as plt


def set_plot_style():
    # academic style for plots
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "font.family": "serif",
            "grid.color": "gray",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            # 'axes.grid': True,
        }
    )

