"""
losses.py - PINN loss functions for orbital dynamics.

Four loss terms work together to train the network:

1. DATA LOSS (L_data)
   Standard MSE between PINN predictions and ground-truth trajectory points.
   Anchors the network to the correct solution.

2. PHYSICS LOSS (L_physics)
   MSE of ODE residuals computed via torch.autograd.grad.  The governing
   equations in normalized coordinates are:

       dx/dt - vx = 0          (position-velocity coupling)
       dy/dt - vy = 0
       dvx/dt + x/r^3 = 0      (Newton's law of gravitation)
       dvy/dt + y/r^3 = 0

   Since we non-dimensionalized with mu->1, the equations are clean.

3. ENERGY LOSS (L_energy)
   Penalizes drift in specific orbital energy epsilon = v^2/2 - 1/r.
   Should be constant for unperturbed Keplerian orbits.

4. ANGULAR MOMENTUM LOSS (L_momentum)
   Penalizes drift in angular momentum L = x*vy - y*vx.
   Should be constant for central force motion (Kepler's 2nd law).
"""

from typing import Optional

import torch


SAFE_RADIUS_EPS = 1e-6


def safe_radius(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Radius guard used anywhere a loss divides by r."""
    return torch.sqrt(x**2 + y**2 + SAFE_RADIUS_EPS)


def compute_data_loss(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    MSE between PINN predictions and ground-truth states.

    Args:
        predicted:    Shape (N, 4) - [x, y, vx, vy]
        ground_truth: Shape (N, 4) - [x, y, vx, vy]
    Returns:
        Scalar MSE loss
    """
    return torch.mean((predicted - ground_truth) ** 2)


def compute_initial_condition_loss(
    predicted_initial: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """MSE at t=0. PINNs need boundary/initial conditions nailed down."""
    return torch.mean((predicted_initial - initial_state) ** 2)


def compute_physics_loss(model, t_colloc: torch.Tensor, x_scale: float) -> torch.Tensor:
    """
    Compute ODE residual loss using automatic differentiation.

    The PINN must satisfy Newton's law of gravitation.  We differentiate
    the network outputs w.r.t. time using autograd, then measure how
    far the derivatives are from satisfying the governing ODEs.

    In normalized coordinates (mu=1):
        residual_x  = dx/dt - vx        -> should be 0
        residual_y  = dy/dt - vy        -> should be 0
        residual_vx = dvx/dt + x/r^3     -> should be 0
        residual_vy = dvy/dt + y/r^3     -> should be 0

    Args:
        model:     PINN model
        t_colloc:  Collocation points, shape (N, 1) - will be set to require grad
        x_scale:   Normalization scale for length [m] (unused in pure Keplerian,
                   kept for API compatibility)
    Returns:
        Scalar MSE of all ODE residuals
    """
    # Ensure gradients flow through time
    t = t_colloc.detach().requires_grad_(True)

    # Forward pass
    state = model(t)                     # (N, 4)
    x  = state[:, 0:1]                   # Keep 2D for autograd
    y  = state[:, 1:2]
    vx = state[:, 2:3]
    vy = state[:, 3:4]

    ones = torch.ones_like(x)

    # Time derivatives via autograd
    # create_graph=True is essential: the physics loss itself must
    # be differentiable so gradients can backpropagate through it.
    dx_dt  = torch.autograd.grad(x,  t, grad_outputs=ones, create_graph=True)[0]
    dy_dt  = torch.autograd.grad(y,  t, grad_outputs=ones, create_graph=True)[0]
    dvx_dt = torch.autograd.grad(vx, t, grad_outputs=ones, create_graph=True)[0]
    dvy_dt = torch.autograd.grad(vy, t, grad_outputs=ones, create_graph=True)[0]

    # Gravitational acceleration in normalized coordinates
    # Pure Keplerian: a = -r/|r|^3  (with mu=1)
    r = safe_radius(x, y)
    r3 = r ** 3

    # ODE residuals (should all be zero)
    res_x  = dx_dt  - vx              # dx/dt = vx
    res_y  = dy_dt  - vy              # dy/dt = vy
    res_vx = dvx_dt + x / r3          # dvx/dt = -x/r^3
    res_vy = dvy_dt + y / r3          # dvy/dt = -y/r^3

    # Mean squared residual across all collocation points
    physics_loss = torch.mean(res_x**2 + res_y**2 + res_vx**2 + res_vy**2)
    return physics_loss


def compute_energy_loss(state: torch.Tensor, target_energy: float = -0.5) -> torch.Tensor:
    """
    Penalize drift in specific orbital energy.

    Normalized energy: epsilon = 0.5(vx^2 + vy^2) - 1/r
    For Keplerian orbits: epsilon = -1/(2a), constant regardless of eccentricity.

    Because length is normalized by the semi-major axis and mu=1, the
    Keplerian target for every scenario here is exactly -0.5.

    Args:
        state: Shape (N, 4) - [x, y, vx, vy] ordered by time
    Returns:
        Scalar MSE of energy drift
    """
    x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    r = safe_radius(x, y)

    kinetic   = 0.5 * (vx**2 + vy**2)
    potential = -1.0 / r
    energy    = kinetic + potential       # Should be constant = -1/(2a)

    # Penalize deviation from the exact normalized Keplerian energy.
    energy_drift = energy - target_energy
    return torch.mean(energy_drift ** 2)


def compute_momentum_loss(
    state: torch.Tensor,
    target_momentum: Optional[float] = None,
) -> torch.Tensor:
    """
    Penalize drift in specific angular momentum (z-component).

    Normalized angular momentum: L = x*vy - y*vx
    Should be constant for central force motion (Kepler's 2nd law).

    Args:
        state: Shape (N, 4) - [x, y, vx, vy] ordered by time
        target_momentum: Exact normalized angular momentum. If omitted,
                         falls back to the first predicted value.
    Returns:
        Scalar MSE of angular momentum drift
    """
    x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

    angular_momentum = x * vy - y * vx   # Should be constant

    if target_momentum is None:
        target = angular_momentum[0]
    else:
        target = torch.as_tensor(
            target_momentum,
            dtype=state.dtype,
            device=state.device,
        )

    momentum_drift = angular_momentum - target
    return torch.mean(momentum_drift ** 2)
