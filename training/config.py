"""
config.py - Physical constants, orbital scenarios, and hyperparameters
for the Physics-Informed Neural Network spacecraft trajectory predictor.

All units are SI unless otherwise noted:
  - Distance: meters [m]
  - Time: seconds [s]
  - Velocity: meters per second [m/s]
  - Mass: kilograms [kg]

Normalization strategy:
  We non-dimensionalize the two-body problem using the semi-major axis (a)
  as the length scale, sqrt(mu/a) as the velocity scale, and sqrt(a^3/mu)
  as the time scale. In these coordinates the gravitational parameter mu = 1,
  and one orbital period maps to 2pi in normalized time. The ODEs become:

      dx/dt = vx          dvx/dt = -x/r^3
      dy/dt = vy          dvy/dt = -y/r^3

  This is the cleanest possible form - no physical constants in the equations.
"""

import dataclasses
import math
from typing import List


# ===================================================================
# Physical Constants
# ===================================================================
G       = 6.67430e-11      # Gravitational constant  [m^3 kg^-1 s^-2]
M_EARTH = 5.972e24         # Earth mass              [kg]
R_EARTH = 6.371e6          # Earth mean radius       [m]
MU      = G * M_EARTH      # Standard grav. param.   [m^3 s^-2]  ~ 3.986e14


# ===================================================================
# Orbital Scenario Definition
# ===================================================================
@dataclasses.dataclass
class OrbitalScenario:
    """
    Defines a Keplerian orbit via periapsis/apoapsis altitudes.
    Derives all orbital elements, normalization scales, and initial conditions.
    """
    name: str
    periapsis_alt_km: float      # Altitude above Earth surface at periapsis [km]
    apoapsis_alt_km: float       # Altitude above Earth surface at apoapsis  [km]
    n_orbits_train: float        # Number of orbits for training time domain
    n_orbits_eval: float         # Number of orbits for evaluation

    # Orbital geometry

    @property
    def r_periapsis(self) -> float:
        """Periapsis radius from Earth center [m]."""
        return R_EARTH + self.periapsis_alt_km * 1e3

    @property
    def r_apoapsis(self) -> float:
        """Apoapsis radius from Earth center [m]."""
        return R_EARTH + self.apoapsis_alt_km * 1e3

    @property
    def semi_major_axis(self) -> float:
        """Semi-major axis a = (r_p + r_a) / 2  [m]."""
        return (self.r_periapsis + self.r_apoapsis) / 2.0

    @property
    def eccentricity(self) -> float:
        """Orbital eccentricity e = (r_a - r_p) / (r_a + r_p)."""
        return (self.r_apoapsis - self.r_periapsis) / (self.r_apoapsis + self.r_periapsis)

    @property
    def period(self) -> float:
        """Orbital period T = 2*pi*sqrt(a^3/mu)  [s]."""
        return 2.0 * math.pi * math.sqrt(self.semi_major_axis**3 / MU)

    @property
    def v_periapsis(self) -> float:
        """Velocity at periapsis via vis-viva: v = sqrt(mu(2/r - 1/a))  [m/s]."""
        return math.sqrt(MU * (2.0 / self.r_periapsis - 1.0 / self.semi_major_axis))

    # Normalization scales
    # These convert physical -> dimensionless coordinates and back.

    @property
    def x_scale(self) -> float:
        """Position normalization: semi-major axis [m]."""
        return self.semi_major_axis

    @property
    def v_scale(self) -> float:
        """Velocity normalization: circular velocity at semi-major axis [m/s]."""
        return math.sqrt(MU / self.semi_major_axis)

    @property
    def t_scale(self) -> float:
        """Time normalization: a / v_scale = sqrt(a^3/mu)  [s].
        One orbit = 2pi in normalized time."""
        return self.semi_major_axis / self.v_scale

    # Initial conditions
    # Spacecraft starts at periapsis on the +x axis, moving in +y.

    @property
    def initial_state(self) -> list:
        """[x0, y0, vx0, vy0] in physical units [m, m, m/s, m/s]."""
        return [self.r_periapsis, 0.0, 0.0, self.v_periapsis]

    @property
    def initial_state_normalized(self) -> list:
        """[x0, y0, vx0, vy0] in dimensionless coordinates."""
        return [
            self.r_periapsis / self.x_scale,   # < 1 for elliptical, = 1 for circular
            0.0,
            0.0,
            self.v_periapsis / self.v_scale,   # > 1 at periapsis for elliptical
        ]

    def __repr__(self) -> str:
        return (
            f"OrbitalScenario('{self.name}', "
            f"alt={self.periapsis_alt_km:.0f}x{self.apoapsis_alt_km:.0f} km, "
            f"e={self.eccentricity:.4f}, "
            f"T={self.period:.0f} s = {self.period/60:.1f} min)"
        )


# ===================================================================
# Pre-defined Orbital Scenarios
# ===================================================================
SCENARIOS = {
    # Circular LEO - ISS-like altitude, simplest case
    "circular_leo": OrbitalScenario(
        name="circular_leo",
        periapsis_alt_km=400.0,
        apoapsis_alt_km=400.0,
        n_orbits_train=5.0,
        n_orbits_eval=10.0,
    ),

    # Elliptical - Artemis II trans-lunar injection parking orbit
    "elliptical": OrbitalScenario(
        name="elliptical",
        periapsis_alt_km=185.0,
        apoapsis_alt_km=2222.0,
        n_orbits_train=5.0,
        n_orbits_eval=10.0,
    ),

    # Highly elliptical - GEO transfer orbit (eccentricity ~ 0.73)
    "highly_elliptical": OrbitalScenario(
        name="highly_elliptical",
        periapsis_alt_km=300.0,
        apoapsis_alt_km=35786.0,
        n_orbits_train=5.0,
        n_orbits_eval=5.0,
    ),
}


# ===================================================================
# Training Hyperparameters
# ===================================================================
@dataclasses.dataclass
class TrainingConfig:
    """All knobs for the training pipeline."""

    # Network architecture
    hidden_dim: int          = 64
    hidden_layers: int       = 4
    num_fourier_freq: int    = 10
    activation: str          = "tanh"    # smooth activation for autograd physics loss

    # Optimizer
    epochs: int              = 15_000
    learning_rate: float     = 1e-3      # Initial LR
    lr_min: float            = 1e-5      # Final LR via CosineAnnealing

    # Data
    n_data_points: int       = 1_000     # Sparse trajectory samples for data loss
    n_collocation: int       = 2_000     # Random points for physics loss per batch
    verlet_steps_per_orbit: int = 10_000 # Integration resolution
    periapsis_collocation_fraction: float = 0.5  # Focus physics checks near perigee
    periapsis_time_sigma: float = 0.2     # Normalized-time std dev around 2*pi*k

    # Loss weights
    lambda_data: float       = 100.0     # Strong trajectory anchor
    lambda_initial: float    = 500.0     # Enforce the initial condition
    lambda_physics: float    = 1.0       # Final ODE residual weight
    lambda_energy: float     = 0.1       # Final energy regularizer
    lambda_momentum: float   = 0.1       # Final angular momentum regularizer
    physics_ramp_start: int  = 1_000
    physics_ramp_end: int    = 4_000
    conservation_ramp_start: int = 3_000
    conservation_ramp_end: int   = 7_000

    # Curriculum learning
    curriculum_stages: list  = dataclasses.field(default_factory=lambda: [
        {"epoch_start": 0,      "n_orbits": 1.0},
        {"epoch_start": 3_000,  "n_orbits": 3.0},
        {"epoch_start": 8_000,  "n_orbits": 5.0},
    ])

    # Logging
    log_interval: int        = 50
    eval_interval: int       = 500

    def get_loss_weights(self, epoch: int) -> dict:
        """
        Scheduled loss weights.

        Strategy:
          - Data and initial-condition weights stay constant.
          - Physics and conservation losses are exactly off at the start.
            Random networks often predict r ~= 0, where x/r^3 is singular.
          - Physics ramps from 0 to its final weight during epochs 1k-4k.
          - Conservation losses ramp from 0 to 0.1 after the physics ramp
            is underway.
        """
        # Strict data pretraining: no physics residuals during the first
        # 1000 epochs. This prevents the r ~= 0 initialization singularity
        # from saturating tanh before the network learns the orbit geometry.
        if epoch < self.physics_ramp_start:
            lp = 0.0
        elif epoch < self.physics_ramp_end:
            span = max(1, self.physics_ramp_end - self.physics_ramp_start)
            t = (epoch - self.physics_ramp_start) / span
            lp = self.lambda_physics * t
        else:
            lp = self.lambda_physics

        # Conservation losses are polish terms. They should not compete with
        # fitting the state before the state itself is physically plausible.
        if epoch < self.conservation_ramp_start:
            le = 0.0
            lm = 0.0
        elif epoch < self.conservation_ramp_end:
            span = max(1, self.conservation_ramp_end - self.conservation_ramp_start)
            t = (epoch - self.conservation_ramp_start) / span
            le = self.lambda_energy * t
            lm = self.lambda_momentum * t
        else:
            le = self.lambda_energy
            lm = self.lambda_momentum

        return {
            "data": self.lambda_data,
            "initial": self.lambda_initial,
            "physics": lp,
            "energy": le,
            "momentum": lm,
        }

    def get_curriculum_n_orbits(self, epoch: int) -> float:
        """Number of orbits to train on at a given epoch (curriculum learning)."""
        n = self.curriculum_stages[0]["n_orbits"]
        for stage in self.curriculum_stages:
            if epoch >= stage["epoch_start"]:
                n = stage["n_orbits"]
        return n
