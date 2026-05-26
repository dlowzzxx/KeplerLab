"""
parametric_dataset.py - Data generation for the generalized Kepler surrogate.

The dataset is generated in normalized two-body coordinates:
    mu = 1
    a = 1
    orbital period = 2*pi

For a selected eccentricity e, the periapsis state is:
    x0 = 1 - e
    y0 = 0
    vx0 = 0
    vy0 = sqrt((1 + e) / (1 - e))

Velocity Verlet is used as the reference integrator, matching the rest of the
project while avoiding large SI-scale numbers during training.
"""

import dataclasses
import math
from typing import Dict, List

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclasses.dataclass
class ParametricDatasetConfig:
    n_train_orbits: int = 72
    n_eval_orbits: int = 8
    n_time_orbits: float = 1.0
    steps_per_orbit: int = 2400
    n_data_batch: int = 4096
    n_collocation: int = 4096
    e_min: float = 0.0
    e_max_train: float = 0.80
    e_max_eval: float = 0.90
    semi_major_axis_min_m: float = 6.6e6
    semi_major_axis_max_m: float = 4.5e7
    periapsis_collocation_fraction: float = 0.5
    periapsis_time_sigma: float = 0.18


def normalized_initial_state(eccentricity: float) -> np.ndarray:
    e = float(np.clip(eccentricity, 0.0, 0.95))
    return np.array([
        1.0 - e,
        0.0,
        0.0,
        math.sqrt((1.0 + e) / max(1e-6, 1.0 - e)),
    ], dtype=np.float32)


def normalized_acceleration(x: float, y: float):
    r_sq = x * x + y * y
    r = math.sqrt(r_sq)
    r_cubed = max(1e-9, r_sq * r)
    return -x / r_cubed, -y / r_cubed


def velocity_verlet_normalized(eccentricity: float, steps_per_orbit: int,
                               n_orbits: float = 1.0):
    dt = (2.0 * math.pi) / steps_per_orbit
    num_steps = int(steps_per_orbit * n_orbits)
    states = np.empty((num_steps + 1, 4), dtype=np.float32)
    times = np.empty(num_steps + 1, dtype=np.float32)

    x, y, vx, vy = normalized_initial_state(eccentricity)
    ax, ay = normalized_acceleration(float(x), float(y))
    states[0] = [x, y, vx, vy]
    times[0] = 0.0

    for i in range(1, num_steps + 1):
        x_new = x + vx * dt + 0.5 * ax * dt * dt
        y_new = y + vy * dt + 0.5 * ay * dt * dt
        ax_new, ay_new = normalized_acceleration(float(x_new), float(y_new))
        vx_new = vx + 0.5 * (ax + ax_new) * dt
        vy_new = vy + 0.5 * (ay + ay_new) * dt

        x, y, vx, vy = x_new, y_new, vx_new, vy_new
        ax, ay = ax_new, ay_new
        states[i] = [x, y, vx, vy]
        times[i] = i * dt

    return times, states


class ParametricOrbitDataset:
    """Precomputed orbit family plus batch samplers for parametric training."""

    def __init__(self, config: ParametricDatasetConfig):
        self.config = config
        self._rng = np.random.default_rng(42)
        self.train_orbits = self._build_train_pool()
        self.eval_cases = self._build_eval_cases()

    def a_to_feature(self, semi_major_axis_m: float) -> float:
        cfg = self.config
        lo = math.log(cfg.semi_major_axis_min_m)
        hi = math.log(cfg.semi_major_axis_max_m)
        return 2.0 * (math.log(semi_major_axis_m) - lo) / (hi - lo) - 1.0

    def feature_to_a(self, feature: float) -> float:
        cfg = self.config
        lo = math.log(cfg.semi_major_axis_min_m)
        hi = math.log(cfg.semi_major_axis_max_m)
        return math.exp(lo + 0.5 * (feature + 1.0) * (hi - lo))

    def get_data_batch(self, e_cap: float):
        eligible = [o for o in self.train_orbits if o["eccentricity"] <= e_cap + 1e-9]
        if not eligible:
            eligible = [self.train_orbits[0]]

        orbit_indices = self._rng.integers(0, len(eligible), size=self.config.n_data_batch)
        times = []
        params = []
        states = []
        for idx in orbit_indices:
            orbit = eligible[int(idx)]
            point_idx = self._rng.integers(0, len(orbit["times"]))
            times.append(orbit["times"][point_idx])
            params.append([orbit["eccentricity"], orbit["a_feature"]])
            states.append(orbit["states"][point_idx])

        return (
            torch.tensor(times, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(params, dtype=torch.float32, device=device),
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
        )

    def get_collocation_batch(self, e_cap: float):
        cfg = self.config
        n_peri = int(cfg.n_collocation * cfg.periapsis_collocation_fraction)
        n_uniform = cfg.n_collocation - n_peri
        t_max = cfg.n_time_orbits * 2.0 * math.pi

        e_uniform = self._rng.uniform(cfg.e_min, e_cap, size=n_uniform)
        a_uniform = self._sample_a(n_uniform)
        t_uniform = self._rng.uniform(0.0, t_max, size=n_uniform)

        e_peri = self._rng.uniform(cfg.e_min, e_cap, size=n_peri)
        a_peri = self._sample_a(n_peri)
        orbit_index = self._rng.integers(0, int(math.floor(cfg.n_time_orbits)) + 1, size=n_peri)
        t_peri = orbit_index * (2.0 * math.pi)
        t_peri = np.clip(
            t_peri + self._rng.normal(0.0, cfg.periapsis_time_sigma, size=n_peri),
            0.0,
            t_max,
        )

        times = np.concatenate([t_uniform, t_peri])
        eccentricities = np.concatenate([e_uniform, e_peri])
        a_values = np.concatenate([a_uniform, a_peri])
        params = np.stack(
            [eccentricities, np.array([self.a_to_feature(a) for a in a_values])],
            axis=1,
        )
        order = self._rng.permutation(len(times))
        return (
            torch.tensor(times[order], dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(params[order], dtype=torch.float32, device=device),
        )

    def get_initial_batch(self, e_cap: float, batch_size: int = 1024):
        e = self._rng.uniform(self.config.e_min, e_cap, size=batch_size)
        a = self._sample_a(batch_size)
        params = np.stack([e, np.array([self.a_to_feature(v) for v in a])], axis=1)
        states = np.array([normalized_initial_state(v) for v in e], dtype=np.float32)
        t0 = np.zeros((batch_size, 1), dtype=np.float32)
        return (
            torch.tensor(t0, dtype=torch.float32, device=device),
            torch.tensor(params, dtype=torch.float32, device=device),
            torch.tensor(states, dtype=torch.float32, device=device),
        )

    def get_eval_case(self, case: Dict):
        times, states = velocity_verlet_normalized(
            case["eccentricity"],
            self.config.steps_per_orbit,
            n_orbits=1.0,
        )
        a_feature = self.a_to_feature(case["semi_major_axis_m"])
        params = np.repeat([[case["eccentricity"], a_feature]], len(times), axis=0)
        return (
            torch.tensor(times, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(params, dtype=torch.float32, device=device),
            torch.tensor(states, dtype=torch.float32, device=device),
        )

    def parameter_bounds(self) -> Dict:
        cfg = self.config
        return {
            "eccentricity": [cfg.e_min, cfg.e_max_train],
            "eccentricity_eval": [cfg.e_min, cfg.e_max_eval],
            "semi_major_axis_m": [
                cfg.semi_major_axis_min_m,
                cfg.semi_major_axis_max_m,
            ],
        }

    def _build_train_pool(self) -> List[Dict]:
        cfg = self.config
        e_values = np.linspace(cfg.e_min, cfg.e_max_train, cfg.n_train_orbits)
        a_values = self._sample_a(cfg.n_train_orbits)
        pool = []
        for e, a in zip(e_values, a_values):
            times, states = velocity_verlet_normalized(
                float(e),
                cfg.steps_per_orbit,
                cfg.n_time_orbits,
            )
            pool.append({
                "eccentricity": float(e),
                "semi_major_axis_m": float(a),
                "a_feature": self.a_to_feature(float(a)),
                "times": times,
                "states": states,
            })
        return pool

    def _build_eval_cases(self) -> List[Dict]:
        cfg = self.config
        a_mid = math.sqrt(cfg.semi_major_axis_min_m * cfg.semi_major_axis_max_m)
        return [
            {"id": "seen_low_e", "label": "Seen low-e", "eccentricity": 0.10,
             "semi_major_axis_m": a_mid, "split": "in_domain"},
            {"id": "seen_mid_e", "label": "Seen mid-e", "eccentricity": 0.45,
             "semi_major_axis_m": a_mid, "split": "in_domain"},
            {"id": "seen_high_e", "label": "Seen high-e", "eccentricity": 0.72,
             "semi_major_axis_m": a_mid, "split": "in_domain"},
            {"id": "ood_extreme_e", "label": "OOD extreme-e", "eccentricity": 0.87,
             "semi_major_axis_m": a_mid, "split": "out_of_domain"},
        ]

    def _sample_a(self, n: int) -> np.ndarray:
        cfg = self.config
        lo = math.log(cfg.semi_major_axis_min_m)
        hi = math.log(cfg.semi_major_axis_max_m)
        return np.exp(self._rng.uniform(lo, hi, size=n))
