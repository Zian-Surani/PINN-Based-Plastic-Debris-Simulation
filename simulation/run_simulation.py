"""One API: explicit experiment and checkpoint, no post-hoc season scaling."""

from core.pinn_inference import load_pinn, predict_frames, DEFAULT_CHECKPOINT


def run_simulation(experiment, times=None, checkpoint=DEFAULT_CHECKPOINT):
    return predict_frames(load_pinn(checkpoint), experiment, times)
