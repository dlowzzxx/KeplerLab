"""
parametric_model.py - Generalized Kepler residual surrogate.

The single-scenario PINNs in KeplerLab learn one orbit at a time. This model is
the separate v2 path: a parameter-conditioned surrogate for a continuous family
of Keplerian orbits.

High-eccentricity motion is not smooth in uniformly sampled time near periapsis,
which made the previous Fourier-only parametric model underfit the GTO case.
This version uses the physically natural coordinate first:

    mean anomaly M -> eccentric anomaly E, where M = E - e sin(E)

The exact normalized two-body state is computed from E, and the neural network
predicts only a small residual on top of that baseline. It is therefore a
hybrid physics-neural surrogate, not a black-box curve fit.
"""

import torch
import torch.nn as nn


class KeplerAnomalyFeatures(nn.Module):
    """Features built from mean anomaly, eccentric anomaly, and orbit shape."""

    def __init__(self, num_frequencies: int = 8, kepler_iterations: int = 7):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.kepler_iterations = kepler_iterations
        freqs = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("frequencies", freqs)

    @property
    def output_dim(self) -> int:
        # sin/cos mean anomaly, sin/cos eccentric anomaly, and 8 shape terms.
        return 4 * self.num_frequencies + 8

    def solve_eccentric_anomaly(
        self,
        mean_anomaly: torch.Tensor,
        eccentricity: torch.Tensor,
    ) -> torch.Tensor:
        """Solve Kepler's equation with differentiable Newton iterations."""
        e = torch.clamp(eccentricity, 0.0, 0.95)

        # A good starter keeps the Newton loop stable for both low-e and high-e
        # cases without introducing discontinuities.
        starter_den = 1.0 - torch.sin(mean_anomaly + e) + torch.sin(mean_anomaly)
        starter_den = torch.clamp(starter_den, min=0.1)
        eccentric_anomaly = mean_anomaly + e * torch.sin(mean_anomaly) / starter_den

        for _ in range(self.kepler_iterations):
            residual = eccentric_anomaly - e * torch.sin(eccentric_anomaly) - mean_anomaly
            slope = torch.clamp(1.0 - e * torch.cos(eccentric_anomaly), min=1e-5)
            eccentric_anomaly = eccentric_anomaly - residual / slope

        return eccentric_anomaly

    def kepler_state(
        self,
        mean_anomaly: torch.Tensor,
        eccentricity: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact normalized [x, y, vx, vy] for a=1 and mu=1."""
        e = torch.clamp(eccentricity, 0.0, 0.95)
        eccentric_anomaly = self.solve_eccentric_anomaly(mean_anomaly, e)
        return self._state_from_anomaly(eccentric_anomaly, e)

    def _state_from_anomaly(
        self,
        eccentric_anomaly: torch.Tensor,
        eccentricity: torch.Tensor,
    ) -> torch.Tensor:
        e = torch.clamp(eccentricity, 0.0, 0.95)
        beta = torch.sqrt(torch.clamp(1.0 - e**2, min=1e-8))
        sin_e = torch.sin(eccentric_anomaly)
        cos_e = torch.cos(eccentric_anomaly)
        denom = torch.clamp(1.0 - e * cos_e, min=1e-5)

        x = cos_e - e
        y = beta * sin_e
        vx = -sin_e / denom
        vy = beta * cos_e / denom
        return torch.cat([x, y, vx, vy], dim=-1)

    def state_and_features(
        self,
        t: torch.Tensor,
        eccentricity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.clamp(eccentricity, 0.0, 0.95)
        eccentric_anomaly = self.solve_eccentric_anomaly(t, e)
        baseline = self._state_from_anomaly(eccentric_anomaly, e)

        mean_args = t * self.frequencies
        eccentric_args = eccentric_anomaly * self.frequencies
        beta = torch.sqrt(torch.clamp(1.0 - e**2, min=1e-8))
        denom = torch.clamp(1.0 - e * torch.cos(eccentric_anomaly), min=1e-5)

        shape_terms = torch.cat(
            [
                baseline,
                denom,
                1.0 / denom,
                beta,
                e,
            ],
            dim=-1,
        )

        features = torch.cat(
            [
                torch.sin(mean_args),
                torch.cos(mean_args),
                torch.sin(eccentric_args),
                torch.cos(eccentric_args),
                shape_terms,
            ],
            dim=-1,
        )
        return baseline, features

    def forward(self, t: torch.Tensor, eccentricity: torch.Tensor) -> torch.Tensor:
        _, features = self.state_and_features(t, eccentricity)
        return features


class ParametricPINN(nn.Module):
    """
    Parameter-conditioned residual surrogate for Keplerian orbit families.

    The condition encoder maps [eccentricity, semi-major-axis feature] into a
    latent vector. The trunk receives anomaly-aware features plus the latent
    condition and predicts a small residual added to the analytic Kepler state.
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        hidden_layers: int = 5,
        num_fourier_freq: int = 8,
        param_dim: int = 2,
        condition_dim: int = 48,
        residual_scale: float = 0.02,
        kepler_iterations: int = 7,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.param_dim = param_dim
        self.condition_dim = condition_dim
        self.residual_scale = residual_scale
        self.output_mode = "kepler_residual"
        self.feature_type = "kepler_anomaly"
        self.fourier = KeplerAnomalyFeatures(num_fourier_freq, kepler_iterations)

        self.condition_encoder = nn.Sequential(
            nn.Linear(param_dim, condition_dim),
            nn.Tanh(),
            nn.Linear(condition_dim, condition_dim),
            nn.Tanh(),
        )

        layers = []
        in_dim = self.fourier.output_dim + condition_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 4))
        self.trunk = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in list(self.condition_encoder) + list(self.trunk):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

        # Start close to the analytic Kepler solution. Training then learns
        # residual corrections only when data or losses justify them.
        final = self.trunk[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def kepler_baseline(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        eccentricity = params[:, 0:1]
        return self.fourier.kepler_state(t, eccentricity)

    def forward(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        eccentricity = params[:, 0:1]
        baseline, features = self.fourier.state_and_features(t, eccentricity)
        condition = self.condition_encoder(params)
        residual = self.trunk(torch.cat([features, condition], dim=-1))
        return baseline + self.residual_scale * residual

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
