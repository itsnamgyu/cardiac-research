"""
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def fill_polar_ax(ax, r1, r2, theta1, theta2, resolution=768, 
                  data=None, cmap=None, norm=None):
    count = (theta2 - theta1) / 2 / np.pi * resolution
    thetas = np.linspace(theta1, theta2, count)
    thetas = np.repeat(thetas[:, np.newaxis], 2, axis=1)

    z = np.ones((thetas.shape[0], 2)) * data

    radii = np.array([r1, r2])
    radii = np.repeat(radii[:, np.newaxis], thetas.shape[0], axis=1).T

    ax.pcolormesh(thetas, radii, z, cmap=cmap, norm=norm)

def plot_six_segment_bullseye(ax, data, segBold=None, cmap=None, norm=None):
    linewidth = 2
    resolution = 768

    try:
        data = data.reshape(-1, 6)
    except(ValueError):
        message = "Could not reshape data matrix into a 6 segment" +\
                  "- n slice matrix"
        raise ValueError(message)


    n_slices = data.shape[0]

    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    thetas = np.linspace(0, 2 * np.pi, resolution)
    radii = np.linspace(0.2, 1, n_slices + 1)

    # Draw circles
    for i in range(radii.shape[0]):
        ax.plot(thetas, np.repeat(radii[i], thetas.shape), '-k', lw=linewidth)

    # Draw bounds for 6 segment slices
    for i in range(6):
        theta = np.pi / 3 * (i + 1)
        ax.plot([theta, theta], [radii[0], 1], '-k', lw=linewidth)

    segment_thetas = np.linspace(np.pi / 3, np.pi * (2 + 1 / 3), 7)

    # Fill the segments
    for i in range(n_slices):
        for j in range(6):
            fill_polar_ax(ax,
                          radii[i],
                          radii[i + 1],
                          segment_thetas[j],
                          segment_thetas[j + 1],
                          data=data[i][j], cmap=cmap, norm=norm)
    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def show_color_bar(fig, data, label=None):
    axl = fig.add_axes([0.05, 0.15, 0.25, 0.05])

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())
    cb = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
                                   orientation='horizontal')
    if label:
        cb.set_label(label)

def display_six_segment_bullseye(data, title=None, unit=None):
    if data.shape[0] == 6 and data.shape[1] != 6:
        print("Warning: (nd_array) 'data' shape should be (n, 6), not (6, n)")

    if data.shape[0] != 6 and data.shape[1] != 6:
        string = "Warning: (nd_array) 'data' shape is {0[0]} by {0[1]}." +\
                 "Use (n, 6) to minimize confusion."
        print(string.format(data.shape))

    fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=1,
            subplot_kw=dict(projection='polar'))

    if title:
        fig.canvas.set_window_title(title)
        ax.set_title(title)

    show_color_bar(fig, data, label=unit)
    plot_six_segment_bullseye(ax, data)

    plt.show()

def main():
    display_six_segment_bullseye(np.random.rand(4, 6), title="Random")

if __name__ == "__main__":
    main()
