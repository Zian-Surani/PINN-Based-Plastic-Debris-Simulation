# hydrology/depth_estimator.py

from config import TERRAIN

MANNING_N = {k: v[2] for k, v in TERRAIN.items()}
SLOPE = {k: v[3] for k, v in TERRAIN.items()}


def estimate_depth(width, velocity, river_type):
    if width <= 0 or velocity <= 0:
        raise ValueError("Width and velocity must be positive SI values")
    n = MANNING_N[river_type]
    s = SLOPE[river_type]
    return (velocity * n / (s**0.5)) ** (3 / 2)
