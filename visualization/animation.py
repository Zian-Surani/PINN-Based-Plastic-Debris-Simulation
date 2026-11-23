# visualization/animation.py

import streamlit as st
import matplotlib.pyplot as plt
import config


def animate(frames, experiment, times):
    placeholder = st.empty()
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots()
        x, y = experiment.grid()
        if frame.shape != (len(x), len(y)) or len(times) != len(frames):
            raise ValueError("Grid/time mismatch")
        ax.pcolormesh(
            x,
            y,
            frame.T,
            cmap=config.COLORMAP,
            shading="auto",
            vmin=frames.min(),
            vmax=frames.max(),
        )
        ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Time = {times[i]:g} s")
        placeholder.pyplot(fig)
        plt.close(fig)
