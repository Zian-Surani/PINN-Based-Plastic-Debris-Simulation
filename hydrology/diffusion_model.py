# hydrology/diffusion_model.py


def compute_diffusion(velocity, depth, beta=0.08):
    if velocity < 0 or depth <= 0 or beta < 0:
        raise ValueError("Invalid dispersion inputs")
    return beta * velocity * depth
