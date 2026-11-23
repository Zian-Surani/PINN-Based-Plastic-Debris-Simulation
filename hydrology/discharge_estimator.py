# hydrology/discharge_estimator.py

from config import TERRAIN, SEASON_DISCHARGE

DISCHARGE_BASE = {name: values[1] for name, values in TERRAIN.items()}
RAIN_SCALE = SEASON_DISCHARGE


def estimate_discharge(river_type, season):
    return DISCHARGE_BASE[river_type] * RAIN_SCALE[season]
