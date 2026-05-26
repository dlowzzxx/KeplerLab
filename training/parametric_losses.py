"""
parametric_losses.py - Losses for the generalized Kepler surrogate.
"""

import torch


SAFE_RADIUS_EPS = 1e-6


def safe_radius(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x**2 + y**2 + SAFE_RADIUS_EPS)


def data_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)


def initial_condition_loss(model, t0: torch.Tensor, params: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
    return torch.mean((model(t0, params) - target) ** 2)


def physics_loss(model, t_colloc: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Enforce the normalized two-body ODE for a parameter-conditioned model.

    The derivative is taken only with respect to time. Orbit parameters are
    treated as conditioning variables, not dynamic states.
    """
    t = t_colloc.detach().requires_grad_(True)
    p = params.detach()
    state = model(t, p)
    x = state[:, 0:1]
    y = state[:, 1:2]
    vx = state[:, 2:3]
    vy = state[:, 3:4]
    ones = torch.ones_like(x)

    dx_dt = torch.autograd.grad(x, t, grad_outputs=ones, create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t, grad_outputs=ones, create_graph=True)[0]
    dvx_dt = torch.autograd.grad(vx, t, grad_outputs=ones, create_graph=True)[0]
    dvy_dt = torch.autograd.grad(vy, t, grad_outputs=ones, create_graph=True)[0]

    r = safe_radius(x, y)
    r3 = r**3
    return torch.mean(
        (dx_dt - vx) ** 2
        + (dy_dt - vy) ** 2
        + (dvx_dt + x / r3) ** 2
        + (dvy_dt + y / r3) ** 2
    )


def energy_loss(state: torch.Tensor) -> torch.Tensor:
    x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    r = safe_radius(x, y)
    energy = 0.5 * (vx**2 + vy**2) - 1.0 / r
    return torch.mean((energy + 0.5) ** 2)


def momentum_loss(state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    eccentricity = params[:, 0]
    target = torch.sqrt(torch.clamp(1.0 - eccentricity**2, min=1e-8))
    angular_momentum = state[:, 0] * state[:, 3] - state[:, 1] * state[:, 2]
    return torch.mean((angular_momentum - target) ** 2)
