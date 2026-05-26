"""
dataset.py - training data generation and normalization for the orbital PINN.

Pipeline:
  1. Generate ground truth with Velocity Verlet in physical SI units.
  2. Normalize time, position, and velocity to dimensionless variables.
  3. Provide sparse trajectory data and physics collocation points.
  4. Convert normalized predictions back to SI units for evaluation/export.
"""

import numpy as np
import torch

from config import OrbitalScenario, TrainingConfig, MU
from integrator import velocity_verlet, compute_orbital_energy, compute_angular_momentum


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OrbitalDataset:
    """Ground-truth trajectory and sampling utilities for one orbit scenario."""

    def __init__(self, scenario: OrbitalScenario, config: TrainingConfig):
        self.scenario = scenario
        self.config = config

        self.x_scale = scenario.x_scale
        self.v_scale = scenario.v_scale
        self.t_scale = scenario.t_scale

        self._generate_ground_truth()
        self._validate_ground_truth()

    def _generate_ground_truth(self):
        """Run Velocity Verlet and normalize the resulting trajectory."""
        scenario = self.scenario
        n_orbits_max = max(scenario.n_orbits_train, scenario.n_orbits_eval)
        total_time = n_orbits_max * scenario.period
        dt = scenario.period / self.config.verlet_steps_per_orbit
        num_steps = int(total_time / dt)

        x0, y0, vx0, vy0 = scenario.initial_state
        self.t_phys, self.states_phys = velocity_verlet(
            x0, y0, vx0, vy0, dt, num_steps, mu=MU
        )

        self.t_norm = self.t_phys / self.t_scale
        self.states_norm = np.empty_like(self.states_phys)
        self.states_norm[:, 0] = self.states_phys[:, 0] / self.x_scale
        self.states_norm[:, 1] = self.states_phys[:, 1] / self.x_scale
        self.states_norm[:, 2] = self.states_phys[:, 2] / self.v_scale
        self.states_norm[:, 3] = self.states_phys[:, 3] / self.v_scale

    def _validate_ground_truth(self):
        """Check that the numerical reference solution is well behaved."""
        energy = compute_orbital_energy(self.states_phys)
        energy_drift = np.max(np.abs(energy - energy[0])) / np.abs(energy[0])

        ang_mom = compute_angular_momentum(self.states_phys)
        mom_drift = np.max(np.abs(ang_mom - ang_mom[0])) / np.abs(ang_mom[0])

        print(
            f"  Ground truth generated: {len(self.t_phys):,} points, "
            f"{self.scenario.n_orbits_eval:.0f} orbits"
        )
        print(f"  Integrator energy drift:   {energy_drift:.2e}")
        print(f"  Integrator momentum drift: {mom_drift:.2e}")

        if energy_drift > 1e-6:
            print(
                f"  WARNING: Energy drift {energy_drift:.2e} is high. "
                f"Consider increasing verlet_steps_per_orbit."
            )

    def get_training_data(self, n_orbits: float):
        """
        Uniformly subsample sparse trajectory data for the supervised loss.

        Returns:
            t_data: normalized time, shape (N, 1)
            states_data: normalized [x, y, vx, vy], shape (N, 4)
        """
        t_max_norm = n_orbits * 2.0 * np.pi
        mask = self.t_norm <= t_max_norm

        t_subset = self.t_norm[mask]
        states_subset = self.states_norm[mask]

        total = len(t_subset)
        n_points = min(self.config.n_data_points, total)
        indices = np.linspace(0, total - 1, n_points, dtype=int)

        t_data = torch.tensor(
            t_subset[indices], dtype=torch.float32, device=device
        ).unsqueeze(-1)
        states_data = torch.tensor(
            states_subset[indices], dtype=torch.float32, device=device
        )

        return t_data, states_data

    def get_collocation_points(self, n_orbits: float) -> torch.Tensor:
        """
        Sample physics collocation times with periapsis emphasis.

        Half of the points are uniform in time. The other half are drawn from
        small Gaussian neighborhoods around periapsis, which occurs at
        normalized times t = 2*pi*k. This gives the ODE residual many more
        checks during the fast high-curvature turn of eccentric orbits.
        """
        t_max_norm = n_orbits * 2.0 * np.pi
        n_total = self.config.n_collocation
        n_periapsis = int(n_total * self.config.periapsis_collocation_fraction)
        n_uniform = n_total - n_periapsis

        t_uniform = torch.rand(n_uniform, 1, device=device) * t_max_norm

        max_k = max(0, int(np.floor(n_orbits)))
        orbit_indices = torch.randint(
            low=0,
            high=max_k + 1,
            size=(n_periapsis, 1),
            device=device,
        ).float()
        periapsis_centers = orbit_indices * (2.0 * np.pi)
        periapsis_noise = (
            torch.randn(n_periapsis, 1, device=device)
            * self.config.periapsis_time_sigma
        )
        t_periapsis = torch.clamp(
            periapsis_centers + periapsis_noise,
            min=0.0,
            max=t_max_norm,
        )

        t_colloc = torch.cat([t_uniform, t_periapsis], dim=0)
        return t_colloc[torch.randperm(t_colloc.shape[0], device=device)]

    def get_eval_data(self):
        """
        Return full-resolution evaluation data, capped at 10,000 samples.
        """
        t_max_norm = self.scenario.n_orbits_eval * 2.0 * np.pi
        mask = self.t_norm <= t_max_norm

        t_sub = self.t_norm[mask]
        states_sub = self.states_norm[mask]

        total = len(t_sub)
        if total > 10_000:
            indices = np.linspace(0, total - 1, 10_000, dtype=int)
            t_sub = t_sub[indices]
            states_sub = states_sub[indices]

        t_eval = torch.tensor(t_sub, dtype=torch.float32, device=device).unsqueeze(-1)
        states_eval = torch.tensor(states_sub, dtype=torch.float32, device=device)

        return t_eval, states_eval

    def denormalize_state(self, state_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized [x, y, vx, vy] tensors back to SI units."""
        scales = torch.tensor(
            [self.x_scale, self.x_scale, self.v_scale, self.v_scale],
            dtype=state_norm.dtype,
            device=state_norm.device,
        )
        return state_norm * scales
