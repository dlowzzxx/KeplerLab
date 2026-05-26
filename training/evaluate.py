"""
evaluate.py - Evaluation metrics and publication-quality plots.

Generates:
  1. Training curves (all loss components vs epoch, log scale)
  2. Trajectory comparison (PINN vs Verlet overlaid, x-y plane)
  3. Accuracy analysis (position error, energy drift, momentum drift over time)
  4. Speed benchmark (PINN inference vs Verlet integration wall-clock time)

All plots use a dark aerospace aesthetic matching the web demo's color scheme.
"""

import time
import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PINN
from dataset import OrbitalDataset
from integrator import velocity_verlet
from losses import safe_radius
from config import OrbitalScenario, MU, R_EARTH


# ===================================================================
# Dark aerospace plot style
# ===================================================================
DARK_STYLE = {
    "figure.facecolor":  "#0a0f1a",
    "axes.facecolor":    "#0a0f1a",
    "axes.edgecolor":    "#334155",
    "text.color":        "#e2e8f0",
    "axes.labelcolor":   "#e2e8f0",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "grid.color":        "#1e293b",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "font.size":         10,
    "legend.facecolor":  "#1e293b",
    "legend.edgecolor":  "#334155",
    "legend.fontsize":   9,
}

# Color palette
C_PINN    = "#34d399"   # Emerald green - PINN predictions
C_VERLET  = "#f97316"   # Orange - Verlet ground truth
C_ERROR   = "#ef4444"   # Red - errors
C_ENERGY  = "#a78bfa"   # Purple - energy
C_MOMTM   = "#fb923c"   # Amber - angular momentum
C_BLUE    = "#60a5fa"   # Blue - general/total
C_EARTH   = "#1e3a5f"   # Dark blue - Earth fill
C_GRID    = "#94a3b8"   # Slate - minor elements


def compute_final_metrics(model: PINN, dataset: OrbitalDataset,
                          scenario: OrbitalScenario) -> dict:
    """
    Compute comprehensive evaluation metrics including speed benchmark.

    Returns a dict with position error, velocity error, conservation
    law drift, and wall-clock timing comparison.
    """
    model.eval()

    with torch.no_grad():
        t_eval, states_gt = dataset.get_eval_data()

        # PINN inference timing
        # Average over 100 runs for stable measurement
        n_runs = 100
        t_start = time.perf_counter()
        for _ in range(n_runs):
            pred = model(t_eval)
        pinn_time = (time.perf_counter() - t_start) / n_runs

        # De-normalize for physical-unit comparison
        pred_phys = dataset.denormalize_state(pred)
        gt_phys   = dataset.denormalize_state(states_gt)

        # Position error [km]
        pos_error = torch.sqrt(
            (pred_phys[:, 0] - gt_phys[:, 0])**2
            + (pred_phys[:, 1] - gt_phys[:, 1])**2
        ) / 1e3

        # Velocity error [m/s]
        vel_error = torch.sqrt(
            (pred_phys[:, 2] - gt_phys[:, 2])**2
            + (pred_phys[:, 3] - gt_phys[:, 3])**2
        )

        # Energy conservation
        x, y, vx, vy = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        r = safe_radius(x, y)
        energy = 0.5 * (vx**2 + vy**2) - 1.0 / r
        energy_drift = (
            torch.max(torch.abs(energy - energy[0])) / torch.abs(energy[0])
        ).item() * 100

        # Angular momentum conservation
        ang_mom = x * vy - y * vx
        mom_drift = (
            torch.max(torch.abs(ang_mom - ang_mom[0])) / torch.abs(ang_mom[0])
        ).item() * 100

    # Verlet integration timing
    x0, y0, vx0, vy0 = scenario.initial_state
    dt = scenario.period / 10_000
    n_steps = int(scenario.n_orbits_eval * scenario.period / dt)

    t_start = time.perf_counter()
    velocity_verlet(x0, y0, vx0, vy0, dt, n_steps, mu=MU)
    verlet_time = time.perf_counter() - t_start

    metrics = {
        "position_error_mean_km": round(float(pos_error.mean()), 4),
        "position_error_max_km":  round(float(pos_error.max()), 4),
        "velocity_error_mean_ms": round(float(vel_error.mean()), 4),
        "velocity_error_max_ms":  round(float(vel_error.max()), 4),
        "energy_drift_max_pct":   round(energy_drift, 6),
        "momentum_drift_max_pct": round(mom_drift, 6),
        "pinn_inference_ms":      round(pinn_time * 1000, 3),
        "verlet_integration_ms":  round(verlet_time * 1000, 3),
        "speedup_factor":         round(verlet_time / max(pinn_time, 1e-9), 1),
        "n_eval_points":          len(t_eval),
    }

    return metrics


def generate_plots(model: PINN, dataset: OrbitalDataset,
                   scenario: OrbitalScenario, history: dict,
                   output_dir: str):
    """
    Generate all evaluation plots with dark aerospace aesthetic.

    Creates:
      - training_curves.png
      - trajectory_{name}.png
      - accuracy_{name}.png
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        t_eval, states_gt = dataset.get_eval_data()
        pred = model(t_eval)
        pred_phys = dataset.denormalize_state(pred)
        gt_phys   = dataset.denormalize_state(states_gt)
        t_np = t_eval.squeeze().cpu().numpy()
    pred_phys_np = pred_phys.cpu().numpy()
    gt_phys_np = gt_phys.cpu().numpy()

    # ===================================================================
    # 1. TRAINING CURVES
    # ===================================================================
    if history.get("epoch"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"TRAINING PROGRESS -- {scenario.name.upper().replace('_', ' ')}",
            fontsize=14, color=C_BLUE, fontweight="bold", y=0.98
        )

        epochs = history["epoch"]

        # Total loss
        axes[0, 0].semilogy(epochs, history["loss_total"],
                            color=C_BLUE, linewidth=1.5, alpha=0.9)
        axes[0, 0].set_title("Total Loss", color=C_BLUE, fontsize=11)
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

        # Data vs Physics loss
        axes[0, 1].semilogy(epochs, history["loss_data"],
                            color=C_PINN, linewidth=1.5, alpha=0.9, label="Data")
        if history.get("loss_initial"):
            axes[0, 1].semilogy(epochs, history["loss_initial"],
                                color=C_GRID, linewidth=1.2, alpha=0.8, label="IC")
        axes[0, 1].semilogy(epochs, history["loss_physics"],
                            color=C_VERLET, linewidth=1.5, alpha=0.9, label="Physics")
        axes[0, 1].set_title("Data vs Physics Loss", color=C_BLUE, fontsize=11)
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Conservation losses
        axes[1, 0].semilogy(epochs, history["loss_energy"],
                            color=C_ENERGY, linewidth=1.5, alpha=0.9, label="Energy")
        axes[1, 0].semilogy(epochs, history["loss_momentum"],
                            color=C_MOMTM, linewidth=1.5, alpha=0.9, label="Momentum")
        axes[1, 0].set_title("Conservation Losses", color=C_BLUE, fontsize=11)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].semilogy(epochs, history["lr"],
                            color=C_GRID, linewidth=1.5, alpha=0.9)
        axes[1, 1].set_title("Learning Rate", color=C_BLUE, fontsize=11)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{output_dir}/training_curves.png",
                    dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
        plt.close()
        print(f"  [OK] training_curves.png")

    # ===================================================================
    # 2. TRAJECTORY COMPARISON
    # ===================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gt_x_km   = gt_phys_np[:, 0] / 1e3
    gt_y_km   = gt_phys_np[:, 1] / 1e3
    pred_x_km = pred_phys_np[:, 0] / 1e3
    pred_y_km = pred_phys_np[:, 1] / 1e3

    # Ground truth (orange, thicker, behind)
    ax.plot(gt_x_km, gt_y_km,
            color=C_VERLET, linewidth=2.5, alpha=0.6, label="Verlet (ground truth)")
    # PINN prediction (green, dashed, on top)
    ax.plot(pred_x_km, pred_y_km,
            color=C_PINN, linewidth=1.5, linestyle="--", alpha=0.9, label="PINN prediction")

    # Draw Earth
    theta = np.linspace(0, 2 * np.pi, 200)
    earth_r = R_EARTH / 1e3
    ax.fill(earth_r * np.cos(theta), earth_r * np.sin(theta),
            color=C_EARTH, alpha=0.8, zorder=5)
    ax.plot(earth_r * np.cos(theta), earth_r * np.sin(theta),
            color=C_BLUE, linewidth=1, alpha=0.6, zorder=6)

    # Start marker
    ax.plot(gt_x_km[0], gt_y_km[0], "o", color="#fbbf24",
            markersize=8, zorder=7, label="Start (periapsis)")

    ax.set_aspect("equal")
    ax.set_xlabel("x (km)", fontsize=11)
    ax.set_ylabel("y (km)", fontsize=11)
    ax.set_title(
        f"ORBITAL TRAJECTORY -- {scenario.name.upper().replace('_', ' ')}",
        fontsize=14, color=C_BLUE, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_{scenario.name}.png",
                dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close()
    print(f"  [OK] trajectory_{scenario.name}.png")

    # ===================================================================
    # 3. ACCURACY ANALYSIS (3-panel)
    # ===================================================================
    t_orbits = t_np / (2.0 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"PINN ACCURACY -- {scenario.name.upper().replace('_', ' ')}",
        fontsize=14, color=C_BLUE, fontweight="bold", y=0.98
    )

    # Position error
    pos_err_km = torch.sqrt(
        (pred_phys[:, 0] - gt_phys[:, 0])**2
        + (pred_phys[:, 1] - gt_phys[:, 1])**2
    ).cpu().numpy() / 1e3

    axes[0].plot(t_orbits, pos_err_km, color=C_ERROR, linewidth=1, alpha=0.8)
    axes[0].fill_between(t_orbits, 0, pos_err_km, color=C_ERROR, alpha=0.15)
    axes[0].set_ylabel("Position Error (km)")
    axes[0].grid(True)

    # Energy conservation
    x, y, vx, vy = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    r = safe_radius(x, y)
    energy = 0.5 * (vx**2 + vy**2) - 1.0 / r
    energy_drift_pct = ((energy - energy[0]) / torch.abs(energy[0]) * 100).cpu().numpy()

    axes[1].plot(t_orbits, energy_drift_pct, color=C_ENERGY, linewidth=1, alpha=0.8)
    axes[1].fill_between(t_orbits, 0, energy_drift_pct, color=C_ENERGY, alpha=0.15)
    axes[1].set_ylabel("Energy Drift (%)")
    axes[1].grid(True)

    # Angular momentum
    ang_mom = x * vy - y * vx
    mom_drift_pct = ((ang_mom - ang_mom[0]) / torch.abs(ang_mom[0]) * 100).cpu().numpy()

    axes[2].plot(t_orbits, mom_drift_pct, color=C_MOMTM, linewidth=1, alpha=0.8)
    axes[2].fill_between(t_orbits, 0, mom_drift_pct, color=C_MOMTM, alpha=0.15)
    axes[2].set_ylabel("Ang. Momentum Drift (%)")
    axes[2].set_xlabel("Orbits")
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/accuracy_{scenario.name}.png",
                dpi=150, bbox_inches="tight", facecolor="#0a0f1a")
    plt.close()
    print(f"  [OK] accuracy_{scenario.name}.png")

    print(f"  All plots saved to {output_dir}/")
