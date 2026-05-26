"""
integrator.py - Velocity Verlet integrator for the two-body problem.

Generates ground-truth trajectories for PINN training and validation.
The same algorithm used in the Artemis II simulator - a symplectic,
second-order integrator that conserves energy to machine precision
over thousands of orbits.

Velocity Verlet scheme:
    x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    a(t+dt) = f(x(t+dt))
    v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt

This is equivalent to the Stormer-Verlet / leapfrog method and
preserves the symplectic structure of Hamiltonian dynamics, meaning
energy errors remain bounded (no secular drift) over arbitrary time.
"""

import numpy as np
from config import MU, R_EARTH


def gravitational_acceleration(x: float, y: float, mu: float = MU):
    """
    Compute pure Newtonian gravitational acceleration at position (x, y).

    a = -mu r / |r|^3

    Args:
        x, y:  Position components [m]
        mu:    Gravitational parameter mu = GM [m^3/s^2]

    Returns:
        ax, ay:  Acceleration components [m/s^2]
    """
    r_sq = x * x + y * y
    r = np.sqrt(r_sq)
    r_cubed = r_sq * r          # r^3 = (r^2)*r - avoids extra sqrt

    ax = -mu * x / r_cubed
    ay = -mu * y / r_cubed

    return ax, ay


def velocity_verlet(x0: float, y0: float, vx0: float, vy0: float,
                    dt: float, num_steps: int, mu: float = MU):
    """
    Integrate the two-body problem using the Velocity Verlet scheme.

    Args:
        x0, y0:     Initial position [m]
        vx0, vy0:   Initial velocity [m/s]
        dt:         Time step [s]
        num_steps:  Number of integration steps
        mu:         Gravitational parameter [m^3/s^2]

    Returns:
        t:       Time array, shape (num_steps + 1,)
        states:  State array [x, y, vx, vy], shape (num_steps + 1, 4)
    """
    # Pre-allocate arrays
    t = np.empty(num_steps + 1)
    states = np.empty((num_steps + 1, 4))

    # Initial conditions
    t[0] = 0.0
    states[0] = [x0, y0, vx0, vy0]

    x, y   = x0, y0
    vx, vy = vx0, vy0
    ax, ay = gravitational_acceleration(x, y, mu)

    half_dt  = 0.5 * dt
    half_dt2 = 0.5 * dt * dt

    for i in range(1, num_steps + 1):
        # Position update
        x_new = x + vx * dt + ax * half_dt2
        y_new = y + vy * dt + ay * half_dt2

        # New acceleration at updated position
        ax_new, ay_new = gravitational_acceleration(x_new, y_new, mu)

        # Velocity update (average of old and new acceleration)
        vx_new = vx + (ax + ax_new) * half_dt
        vy_new = vy + (ay + ay_new) * half_dt

        # Store
        t[i] = i * dt
        states[i] = [x_new, y_new, vx_new, vy_new]

        # Advance
        x,  y  = x_new,  y_new
        vx, vy = vx_new, vy_new
        ax, ay = ax_new, ay_new

    return t, states


def compute_orbital_energy(states: np.ndarray, mu: float = MU) -> np.ndarray:
    """
    Compute specific orbital energy at each time step.

    epsilon = v^2/2 - mu/r    [J/kg = m^2/s^2]

    For a Keplerian orbit this should be constant: epsilon = -mu/(2a).
    """
    x, y   = states[:, 0], states[:, 1]
    vx, vy = states[:, 2], states[:, 3]
    r = np.sqrt(x**2 + y**2)
    return 0.5 * (vx**2 + vy**2) - mu / r


def compute_angular_momentum(states: np.ndarray) -> np.ndarray:
    """
    Compute specific angular momentum (z-component) at each time step.

    L = x*vy - y*vx    [m^2/s]

    For a central force this should be constant (Kepler's second law).
    """
    x, y   = states[:, 0], states[:, 1]
    vx, vy = states[:, 2], states[:, 3]
    return x * vy - y * vx


def validate_integrator(scenario, steps_per_orbit: int = 10_000, n_orbits: int = 10):
    """
    Validate integrator accuracy by checking energy and momentum conservation.

    Returns max relative drift in energy and angular momentum over n_orbits.
    """
    x0, y0, vx0, vy0 = scenario.initial_state
    dt = scenario.period / steps_per_orbit
    num_steps = int(n_orbits * steps_per_orbit)

    _, states = velocity_verlet(x0, y0, vx0, vy0, dt, num_steps, mu=MU)

    energy = compute_orbital_energy(states)
    ang_mom = compute_angular_momentum(states)

    energy_drift = np.max(np.abs(energy - energy[0])) / np.abs(energy[0])
    mom_drift    = np.max(np.abs(ang_mom - ang_mom[0])) / np.abs(ang_mom[0])

    return energy_drift, mom_drift
