"""
export.py - Export trained PINN weights as JSON for browser inference.

The exported JSON contains everything the JavaScript web demo needs:
  1. Network architecture parameters
  2. Layer weights and biases (as nested arrays)
  3. Normalization scales (to convert PINN output -> physical units)
  4. Orbital scenario metadata
  5. Training metrics for the dashboard
  6. Training history (loss curves)

The JS demo will reconstruct the MLP from these weights and run
forward inference with pure matrix multiplication + tanh - no ML
framework required in the browser.
"""

import json
import os
import torch
import numpy as np

from model import PINN
from config import OrbitalScenario


def export_model(model: PINN, scenario: OrbitalScenario,
                 metrics: dict, history: dict, output_path: str) -> dict:
    """
    Export trained model as JSON for browser deployment.

    Args:
        model:       Trained PINN model
        scenario:    Orbital scenario used for training
        metrics:     Final evaluation metrics dict
        history:     Training history dict
        output_path: Path to write JSON file

    Returns:
        The exported data dict
    """
    # Architecture metadata
    model_data = {
        "architecture": {
            "input_dim": model.fourier.output_dim,
            "hidden_dim": model.network[0].out_features,
            "hidden_layers": sum(1 for m in model.network if isinstance(m, torch.nn.Linear)) - 1,
            "output_dim": 4,
            "activation": "tanh",
            "num_fourier_frequencies": model.fourier.num_frequencies,
        },

        # Layer weights
        "weights": {},

        # Normalization scales
        "normalization": {
            "t_scale": scenario.t_scale,
            "x_scale": scenario.x_scale,
            "v_scale": scenario.v_scale,
        },

        # Scenario metadata
        "scenario": {
            "name": scenario.name,
            "periapsis_alt_km": scenario.periapsis_alt_km,
            "apoapsis_alt_km": scenario.apoapsis_alt_km,
            "period_s": scenario.period,
            "semi_major_axis_m": scenario.semi_major_axis,
            "eccentricity": scenario.eccentricity,
            "initial_state": scenario.initial_state,
            "initial_state_normalized": scenario.initial_state_normalized,
            "r_earth_m": 6.371e6,
        },

        # Final metrics
        "metrics": metrics,

        # Training history (for loss curve visualization)
        "training_history": {
            "epochs": history.get("epoch", []),
            "loss_total": history.get("loss_total", []),
            "loss_data": history.get("loss_data", []),
            "loss_initial": history.get("loss_initial", []),
            "loss_physics": history.get("loss_physics", []),
            "loss_energy": history.get("loss_energy", []),
            "loss_momentum": history.get("loss_momentum", []),
        },
    }

    # Extract weights from Sequential model
    layer_idx = 0
    for module in model.network:
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias   = module.bias.detach().cpu().numpy()

            model_data["weights"][f"layer_{layer_idx}"] = {
                "weight": weight.tolist(),
                "bias":   bias.tolist(),
            }
            layer_idx += 1

    # Write JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(model_data, f, separators=(",", ":"))  # Compact format

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [OK] Exported to {output_path}")
    print(f"    Size: {size_kb:.1f} KB")
    print(f"    Layers: {layer_idx}")
    print(f"    Fourier frequencies: {model.fourier.num_frequencies}")

    return model_data


def _linear_layers(module):
    """Serialize Linear layers from a Sequential module in execution order."""
    layers = []
    for item in module:
        if isinstance(item, torch.nn.Linear):
            layers.append({
                "weight": item.weight.detach().cpu().numpy().tolist(),
                "bias": item.bias.detach().cpu().numpy().tolist(),
            })
    return layers


def export_parametric_model(model, dataset_config, metrics: dict,
                            history: dict, output_path: str) -> dict:
    """
    Export the generalized Kepler surrogate as browser-readable JSON.

    The browser uses this to evaluate Model(t, e, a) directly and compare it
    against a fresh Velocity Verlet integration for arbitrary slider-selected
    orbits inside the training bounds.
    """
    model_data = {
        "model_type": "parametric_kepler_residual_surrogate",
        "architecture": {
            "time_input_dim": model.fourier.output_dim,
            "param_dim": model.param_dim,
            "condition_dim": model.condition_dim,
            "hidden_dim": model.hidden_dim,
            "hidden_layers": model.hidden_layers,
            "output_dim": 4,
            "activation": "tanh",
            "num_fourier_frequencies": model.fourier.num_frequencies,
            "feature_type": getattr(model, "feature_type", "fourier_time"),
            "output_mode": getattr(model, "output_mode", "direct_state"),
            "residual_scale": getattr(model, "residual_scale", 1.0),
            "kepler_iterations": getattr(model.fourier, "kepler_iterations", 0),
        },
        "parameter_bounds": {
            "eccentricity": [
                dataset_config.e_min,
                dataset_config.e_max_train,
            ],
            "eccentricity_eval": [
                dataset_config.e_min,
                dataset_config.e_max_eval,
            ],
            "semi_major_axis_m": [
                dataset_config.semi_major_axis_min_m,
                dataset_config.semi_major_axis_max_m,
            ],
        },
        "normalization": {
            "mu_normalized": 1.0,
            "semi_major_axis_normalized": 1.0,
            "period_normalized": 2.0 * np.pi,
            "state_units": "normalized [x, y, vx, vy]",
        },
        "weights": {
            "condition_encoder": _linear_layers(model.condition_encoder),
            "trunk": _linear_layers(model.trunk),
        },
        "metrics": metrics,
        "training_history": history,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model_data, f, separators=(",", ":"))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [OK] Exported parametric model to {output_path}")
    print(f"    Size: {size_kb:.1f} KB")
    print(f"    Parameters: {model.count_parameters():,}")
    return model_data
