# hydrology/river_parameters.py
from config import TERRAIN


def get_width_range(river_type):
    width = TERRAIN[river_type][0]
    return (width, width)


def get_climate_multiplier(season):
    raise RuntimeError(
        "Post-inference concentration scaling is retired. Use config.make_experiment: season scales discharge only."
    )
