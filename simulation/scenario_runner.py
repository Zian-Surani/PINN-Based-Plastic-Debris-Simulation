"""Scenario evaluation uses the same canonical configuration as training."""

from config import all_experiments
from core.pinn_inference import load_pinn, DEFAULT_CHECKPOINT
from simulation.validation_suite import evaluate


def run_scenarios(checkpoint=DEFAULT_CHECKPOINT):
    model = load_pinn(checkpoint)
    return [evaluate(model, e)["metrics"] for e in all_experiments()]
