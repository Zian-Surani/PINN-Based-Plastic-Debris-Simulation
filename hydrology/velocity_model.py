# hydrology/velocity_model.py


def compute_velocity(discharge, width, depth):
    if discharge < 0 or width <= 0 or depth <= 0:
        raise ValueError("Invalid discharge/channel geometry")
    return discharge / (width * depth)
