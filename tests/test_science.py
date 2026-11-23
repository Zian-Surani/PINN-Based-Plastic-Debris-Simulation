import unittest, tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from config import (
    Experiment,
    make_experiment,
    all_experiments,
    SCHEMA_VERSION,
)
from simulation.finite_volume import solve, timestep_limit
from core.pinn_model import PINN, derivative, residual
from core.pinn_inference import load_pinn, predict_points
from simulation.accumulation_metrics import compute_accumulation
from simulation.observational_validation import (
    normalize_observation_table,
    evaluate_transect_observations,
)
from simulation.uncertainty_quantification import residual_interval_audit


class ScienceTests(unittest.TestCase):
    def test_single_baseline_definition(self):
        self.assertEqual(Experiment(), make_experiment())

    def test_scenario_grid_and_units(self):
        for e in all_experiments():
            x, y = e.grid()
            self.assertEqual((len(x), len(y)), (200, 23))
            self.assertAlmostEqual(x[0], e.dx / 2)
            self.assertEqual(e.horizon, 300)
            self.assertGreater(e.diffusion, 0)

    def test_invalid_experiment(self):
        for kw in (
            {"u": -1},
            {"diffusion": -1},
            {"horizon": 0},
            {"nx": 2.5},
            {"width": float("nan")},
        ):
            with self.assertRaises(ValueError):
                Experiment(**kw)

    def test_combined_bound(self):
        e = make_experiment()
        dt = timestep_limit(e)
        self.assertAlmostEqual(
            dt * (e.u / e.dx + e.diffusion * (3 / e.dx**2 + 2 / e.dy**2)), 0.8
        )

    def test_conservative_open_flux(self):
        e = make_experiment(nx=50, ny=12)
        r = solve(e)
        self.assertLess(r["relative_balance_error"], 1e-12)
        self.assertGreaterEqual(r["minimum_density"], 0)
        self.assertGreater(r["escaped_mass_kg"][-1], 0)

    def test_diffusion_inlet_export_accounted(self):
        e = make_experiment(nx=40, ny=12, u=0)
        r = solve(e)
        np.testing.assert_allclose(
            r["mass_kg"] + r["escaped_mass_kg"], r["mass_kg"][0], rtol=1e-12
        )
        self.assertGreaterEqual(r["escaped_mass_kg"][-1], 0)

    def test_invalid_solver_inputs(self):
        e = make_experiment(nx=10, ny=5)
        for times in ([1, 2], [0, 0], [0, -1], [0, 301]):
            with self.assertRaises(ValueError):
                solve(e, times=times)
        with self.assertRaises(ValueError):
            solve(e, initial=-np.ones((10, 5)))

    def test_diffusion_exact_solution_refines(self):
        errors = []
        for n in (20, 40):
            e = make_experiment(
                length=20,
                width=10,
                horizon=2,
                u=0,
                diffusion=0.2,
                nx=n,
                ny=n // 2,
                n_times=2,
            )
            x, y = e.grid()
            mode = np.sin(np.pi * x[:, None] / 40) * np.ones((1, e.ny))
            c0 = 0.001 * mode
            exact = c0 * np.exp(-0.2 * np.pi**2 / 1600 * 2)
            r = solve(e, initial=c0)
            errors.append(np.sqrt(np.mean((r["frames"][-1] - exact) ** 2)))
        self.assertLess(errors[1], errors[0] / 2)

    def test_neural_positivity_and_hard_boundaries(self):
        model = PINN()
        e = make_experiment()
        z = torch.rand(40, 6)
        z[:, 3:] = torch.tensor(e.coefficients)
        z[:10, 0] = 0
        z[10:20, 0] = 1
        z[20:30, 1] = 0
        z[30:, 1] = 1
        z.requires_grad_(True)
        c = model(z)
        g = derivative(c, z)
        self.assertTrue(bool((c >= 0).all()))
        torch.testing.assert_close(c[:10], torch.zeros(10, 1), atol=1e-7, rtol=0)
        self.assertLess(float(g[10:20, 0].detach().abs().max()), 1e-6)
        self.assertLess(float(g[20:, 1].detach().abs().max()), 1e-6)

    def test_residual_chain_rule(self):
        class Exact(torch.nn.Module):
            def forward(self, z):
                return z[:, 0:1] ** 2 + z[:, 1:2] ** 2 + z[:, 2:3]

        z = torch.rand(30, 6)
        expected = 1 + 2 * z[:, 3:4] * z[:, 0:1] - 2 * z[:, 4:5] - 2 * z[:, 5:6]
        torch.testing.assert_close(residual(Exact(), z), expected)

    def test_checkpoint_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "old.pt"
            torch.save(PINN().state_dict(), p)
            with self.assertRaises(ValueError):
                load_pinn(p)
            e = make_experiment()
            m = PINN()
            torch.save(
                {
                    "metadata": {
                        "schema": SCHEMA_VERSION,
                        "width": 64,
                        "signatures": [e.signature()],
                    },
                    "state_dict": m.state_dict(),
                },
                p,
            )
            m = load_pinn(p)
            with self.assertRaises(ValueError):
                predict_points(m, replace(e, u=e.u * 1.1), 0.0, 0.0, 0.0)
            with self.assertRaises(ValueError):
                predict_points(m, e, -1.0, 0.0, 0.0)

    def test_time_integral_not_mean(self):
        a = np.ones((3, 2, 2)) * 2
        np.testing.assert_allclose(compute_accumulation(a, [0, 1, 3]), 6)

    def test_observation_units_and_registration(self):
        with self.assertRaises(ValueError):
            normalize_observation_table(
                pd.DataFrame({"distance": [1, 2, 3], "obs": [1, 2, 3]})
            )
        obs = pd.DataFrame(
            {
                "distance_m": [10, 11, 12],
                "time_s": [0, 0, 0],
                "density_kg_m2": [1, 1, 1],
            }
        )
        with self.assertRaises(ValueError):
            evaluate_transect_observations(
                obs, [0, 1], [0, 1], [0, 1], np.ones((2, 2, 2))
            )

    def test_residual_intervals_reject_bad_split(self):
        with self.assertRaises(ValueError):
            residual_interval_audit(
                np.zeros((4, 2, 2)), np.zeros((4, 2, 2)), calibration_fraction=0.99
            )

    def test_seeded_initialization(self):
        torch.manual_seed(7)
        a = PINN()
        torch.manual_seed(7)
        b = PINN()
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y)

    def test_advection_diffusion_analytic_refinement(self):
        from scripts.extended_verification import analytic_case

        coarse, fine = analytic_case(20), analytic_case(40)
        self.assertLess(coarse["relative_l2"], 0.005)
        self.assertLess(fine["relative_l2"], coarse["relative_l2"] * 0.6)
        self.assertLess(fine["relative_mass_balance"], 1e-12)

    def test_research_interpolation_does_not_unlock_production(self):
        from dataclasses import asdict
        from config import all_experiments
        from core.pinn_inference import validate_experiment
        from scripts.extended_verification import unseen_experiments

        model = PINN()
        model.metadata = dict(
            experiments=[asdict(e) for e in all_experiments()],
            signatures=[e.signature() for e in all_experiments()],
        )
        unseen = unseen_experiments()[0]
        with self.assertRaises(ValueError):
            validate_experiment(model, unseen)
        validate_experiment(model, unseen, research=True)
        with self.assertRaises(ValueError):
            validate_experiment(model, replace(unseen, horizon=600), research=True)
        with self.assertRaises(ValueError):
            validate_experiment(model, replace(unseen, u=100), research=True)

    def test_interval_undercoverage_is_explicit(self):
        pred = np.zeros((6, 2, 2))
        ref = pred.copy()
        ref[3:] = 1
        result = residual_interval_audit(pred, ref)
        self.assertFalse(result["nominal_coverage_met_on_evaluation"])
        self.assertFalse(result["guaranteed_coverage"])
        self.assertEqual(result["empirical_coverage"], 0)

    def test_mass_diagnostic_detects_unbalanced_loss(self):
        from config import initial_field
        from simulation.validation_suite import compute_mass_metrics

        exp = make_experiment()
        initial = initial_field(exp)
        total = initial.sum() * exp.dx * exp.dy
        times = np.array([0.0, 1.0, 2.0])
        frames = np.array([initial, initial * 0.9, initial * 0.8])
        balanced = compute_mass_metrics(frames, exp, times, np.full(3, 0.1 * total))
        self.assertLess(balanced["max_relative_balance_error"], 1e-12)
        missing_export = compute_mass_metrics(frames, exp, times, np.zeros(3))
        self.assertAlmostEqual(missing_export["max_relative_balance_error"], 0.2)


if __name__ == "__main__":
    unittest.main()
