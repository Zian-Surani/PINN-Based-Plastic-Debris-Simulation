# hydrology/width_estimator.py

from config import TERRAIN


def estimate_width(river_type):
    return TERRAIN[river_type][0]
