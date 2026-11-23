import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_animation(frames, path, experiment, times):
    fig, ax = plt.subplots()
    if frames.shape != (len(times), experiment.nx, experiment.ny):
        raise ValueError("Grid/time mismatch")
    im = ax.imshow(
        frames[0].T,
        origin="lower",
        aspect="auto",
        extent=[0, experiment.length, 0, experiment.width],
        vmin=frames.min(),
        vmax=frames.max(),
    )
    ax.set(xlabel="x (m)", ylabel="y (m)")

    def update(i):
        im.set_data(frames[i].T)
        ax.set_title(f"Time = {times[i]:g} s")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=300)

    ani.save(path, writer="ffmpeg", dpi=200)
    plt.close(fig)
