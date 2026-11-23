"""Positive, parameter-conditioned hybrid PINN. All inputs dimensionless."""

import torch
from torch import nn
import torch.nn.functional as F


class PINN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(7, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        nn.init.constant_(self.net[-1].bias, -3.0)

    def forward(self, z):
        # xi, eta, tau, a=uT/L, bx=DT/L^2, by=DT/W^2
        xi, eta, t, a, bx, by = z.split(1, dim=-1)
        # Smooth boundary-flattened coordinate: s'(0)=s'(1)=0.
        # Dispersion/advection boundary-layer length divided by domain length.
        e = (bx / a.clamp_min(1e-12)).clamp(1e-5, 0.02)
        end = torch.exp(-1 / e)
        s = (
            (1 + end) * xi
            + e * (torch.exp(-xi / e) - 1)
            - e * (torch.exp(-(1 - xi) / e) - end)
        ) / ((1 + end) - 2 * e * (1 - end))
        f = torch.cat(
            (
                2 * s - 1,
                2 * (s - a * t) - 1,
                torch.cos(torch.pi * eta),
                2 * t - 1,
                a / 5,
                bx / 0.001,
                by,
            ),
            dim=-1,
        )
        # Zero inlet, unit outlet and zero outlet derivative; nearly one inside.
        inlet = (-torch.expm1(-xi / e) - xi * end / e) / (1 - end - end / e)
        return inlet * F.softplus(self.net(f))


def derivative(c, z):
    return torch.autograd.grad(c, z, torch.ones_like(c), create_graph=True)[0]


def residual(model, z):
    z = z.detach().requires_grad_(True)
    c = model(z)
    g = derivative(c, z)
    xx = derivative(g[:, 0:1], z)[:, 0:1]
    yy = derivative(g[:, 1:2], z)[:, 1:2]
    return g[:, 2:3] + z[:, 3:4] * g[:, 0:1] - z[:, 4:5] * xx - z[:, 5:6] * yy
