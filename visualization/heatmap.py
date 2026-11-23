# visualization/heatmap.py

import matplotlib.pyplot as plt
import config


def plot_heatmap(field, title, experiment):
    fig, ax = plt.subplots()
    x, y = experiment.grid()
    if field.shape != (len(x), len(y)):
        raise ValueError("Expected (x,y) field")
    im = ax.pcolormesh(x, y, field.T, cmap=config.COLORMAP, shading="auto")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Tracer surface density (kg/m²)")
    return fig
