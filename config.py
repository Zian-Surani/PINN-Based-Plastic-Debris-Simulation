"""Single experiment definition. SI units, local coordinates, cell-centred grids."""

from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
import numpy as np

SCHEMA_VERSION = 5
NX, NY = 200, 23
N_TIME_STEPS = 61  # output times, including t=0
TIME_MAX = 300.0  # seconds; former 5 s results are superseded
COLORMAP = "viridis"
SEEDS = (2026, 2027, 2028)
# Idealized scenarios, not measured site hydraulics. Widths are midpoints of
# GRWL-informed global ranges; see docs/DATA_PROVENANCE.md. Other hydraulics
# and discharge multipliers are prescribed assumptions, not raster extraction.
TERRAIN = {
    "plain": (118.8, 1200.0, 0.030, 0.001),
    "mountain": (52.8, 300.0, 0.035, 0.010),
    "urban": (81.0, 600.0, 0.025, 0.002),
    "coastal": (335.5, 2000.0, 0.040, 0.0005),
}
SEASON_DISCHARGE = {"dry": 0.7, "normal": 1.0, "wet": 1.3}


@dataclass(frozen=True)
class Experiment:
    terrain: str = "plain"
    season: str = "normal"
    length: float = 1000.0
    width: float = 118.8
    horizon: float = TIME_MAX
    nx: int = NX
    ny: int = NY
    n_times: int = N_TIME_STEPS
    u: float = 1200 / (118.8 * (1200 * 0.03 / (118.8 * math.sqrt(0.001))) ** (0.6))
    diffusion: float = 0.08 * u * (1200 * 0.03 / (118.8 * math.sqrt(0.001))) ** (0.6)
    initial_density: float = 0.001  # kg/m^2; synthetic depth-integrated tracer
    boundary: str = "zero_inlet_concentration_open_outlet_reflecting_walls"
    initial: str = "compact_pulse_v1"

    def __post_init__(self):
        for name in ("length", "width", "horizon", "initial_density"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.u)
            or self.u < 0
            or not math.isfinite(self.diffusion)
            or self.diffusion < 0
        ):
            raise ValueError(
                "Boundary model requires u>=0 and D>=0; lateral velocity is zero."
            )
        if any(type(v) is not int or v < 2 for v in (self.nx, self.ny, self.n_times)):
            raise ValueError("Grid/time counts must be integers >=2")
        if (
            self.boundary != "zero_inlet_concentration_open_outlet_reflecting_walls"
            or self.initial != "compact_pulse_v1"
        ):
            raise ValueError("Unsupported boundary or initial condition")

    @property
    def dx(self):
        return self.length / self.nx

    @property
    def dy(self):
        return self.width / self.ny

    @property
    def coefficients(self):
        return (
            self.u * self.horizon / self.length,
            self.diffusion * self.horizon / self.length**2,
            self.diffusion * self.horizon / self.width**2,
        )

    def grid(self):
        return (
            (np.arange(self.nx) + 0.5) * self.dx,
            (np.arange(self.ny) + 0.5) * self.dy,
        )

    def times(self):
        return np.linspace(0, self.horizon, self.n_times)

    def physical_dict(self):
        return {
            k: v for k, v in asdict(self).items() if k not in ("nx", "ny", "n_times")
        }

    def signature(self):
        return hashlib.sha256(
            json.dumps(self.physical_dict(), sort_keys=True).encode()
        ).hexdigest()

    def refined(self, nx, ny):
        return replace(self, nx=nx, ny=ny)


def make_experiment(terrain="plain", season="normal", **overrides):
    if terrain not in TERRAIN or season not in SEASON_DISCHARGE:
        raise ValueError("Unknown terrain or season")
    width, q0, n, slope = TERRAIN[terrain]
    q = q0 * SEASON_DISCHARGE[season]
    # Wide rectangular Manning approximation: Q=(W/n) h^(5/3) sqrt(S).
    depth = (q * n / (width * math.sqrt(slope))) ** (3 / 5)
    u = q / (width * depth)
    values = dict(
        terrain=terrain, season=season, width=width, u=u, diffusion=0.08 * u * depth
    )
    values.update(overrides)
    return Experiment(**values)


def all_experiments():
    return [make_experiment(t, s) for t in TERRAIN for s in SEASON_DISCHARGE]


def initial_normalized(x, y):
    """Compact smooth longitudinal pulse; zero normal wall derivative."""
    r = (np.asarray(x) - 0.25) / 0.15
    inside = np.abs(r) < 1
    safe = np.where(inside, 1 - r * r, 1.0)
    return np.where(inside, np.exp(1 - 1 / safe), 0.0) * (
        1 + 0.3 * np.cos(2 * np.pi * np.asarray(y))
    )


def initial_field(exp):
    x, y = exp.grid()
    return exp.initial_density * initial_normalized(
        x[:, None] / exp.length, y[None, :] / exp.width
    )
