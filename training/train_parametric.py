"""
train_parametric.py - Train the generalized Kepler surrogate.

This is KeplerLab v2's optional training path. It does not replace the
single-scenario Failure Lab models. It produces:

    models/parametric_kepler.json
    web/models/parametric_kepler.json

The browser Surrogate Solver tab will remain in a model-pending state until
this file is trained and exported.
"""

import argparse
import dataclasses
import os
import shutil
import time

import torch
import torch.optim as optim

from export import export_parametric_model
from parametric_dataset import ParametricDatasetConfig, ParametricOrbitDataset
from parametric_losses import (
    data_loss,
    energy_loss,
    initial_condition_loss,
    momentum_loss,
    physics_loss,
    safe_radius,
)
from parametric_model import ParametricPINN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclasses.dataclass
class ParametricTrainingConfig:
    epochs: int = 20_000
    hidden_dim: int = 96
    hidden_layers: int = 5
    num_fourier_freq: int = 8
    learning_rate: float = 8e-4
    lr_min: float = 1e-5
    lambda_data: float = 100.0
    lambda_initial: float = 500.0
    lambda_physics: float = 1.0
    lambda_energy: float = 0.1
    lambda_momentum: float = 0.1
    physics_ramp_start: int = 1200
    physics_ramp_end: int = 5000
    conservation_ramp_start: int = 4000
    conservation_ramp_end: int = 9000
    e_curriculum_start: float = 0.10
    log_interval: int = 100
    eval_interval: int = 1000

    def loss_weights(self, epoch: int) -> dict:
        if epoch < self.physics_ramp_start:
            lp = 0.0
        elif epoch < self.physics_ramp_end:
            span = self.physics_ramp_end - self.physics_ramp_start
            lp = self.lambda_physics * (epoch - self.physics_ramp_start) / span
        else:
            lp = self.lambda_physics

        if epoch < self.conservation_ramp_start:
            le = 0.0
            lm = 0.0
        elif epoch < self.conservation_ramp_end:
            span = self.conservation_ramp_end - self.conservation_ramp_start
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

    def eccentricity_cap(self, epoch: int, e_max: float) -> float:
        warm = max(1, int(0.70 * self.epochs))
        if epoch >= warm:
            return e_max
        t = epoch / warm
        return min(e_max, self.e_curriculum_start + (e_max - self.e_curriculum_start) * t)


def evaluate(model: ParametricPINN, dataset: ParametricOrbitDataset) -> dict:
    model.eval()
    cases = []
    with torch.no_grad():
        for case in dataset.eval_cases:
            t_eval, params_eval, states_gt = dataset.get_eval_case(case)
            pred = model(t_eval, params_eval)
            a = case["semi_major_axis_m"]
            pos_error_norm = torch.sqrt(
                (pred[:, 0] - states_gt[:, 0]) ** 2
                + (pred[:, 1] - states_gt[:, 1]) ** 2
            )
            pos_error_km = pos_error_norm * (a / 1000.0)

            x, y, vx, vy = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            r = safe_radius(x, y)
            energy = 0.5 * (vx**2 + vy**2) - 1.0 / r
            energy_drift = (
                torch.max(torch.abs(energy + 0.5)) / 0.5
            ).item() * 100

            target_h = (1.0 - case["eccentricity"] ** 2) ** 0.5
            h = x * vy - y * vx
            momentum_drift = (
                torch.max(torch.abs(h - target_h)) / max(1e-8, abs(target_h))
            ).item() * 100

            cases.append({
                "id": case["id"],
                "label": case["label"],
                "split": case["split"],
                "eccentricity": case["eccentricity"],
                "position_error_mean_km": round(float(pos_error_km.mean()), 4),
                "position_error_max_km": round(float(pos_error_km.max()), 4),
                "energy_drift_max_pct": round(float(energy_drift), 6),
                "momentum_drift_max_pct": round(float(momentum_drift), 6),
            })

    in_domain = [c for c in cases if c["split"] == "in_domain"]
    ood = [c for c in cases if c["split"] == "out_of_domain"]
    return {
        "cases": cases,
        "in_domain_mean_error_km": round(
            sum(c["position_error_mean_km"] for c in in_domain) / len(in_domain),
            4,
        ),
        "out_of_domain_mean_error_km": round(
            sum(c["position_error_mean_km"] for c in ood) / max(1, len(ood)),
            4,
        ),
    }


def train(config: ParametricTrainingConfig, dataset_config: ParametricDatasetConfig,
          export: bool):
    dataset = ParametricOrbitDataset(dataset_config)
    model = ParametricPINN(
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        num_fourier_freq=config.num_fourier_freq,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr_min,
    )
    history = {
        "epoch": [],
        "loss_total": [],
        "loss_data": [],
        "loss_initial": [],
        "loss_physics": [],
        "loss_energy": [],
        "loss_momentum": [],
        "eccentricity_cap": [],
        "lr": [],
    }

    best_loss = float("inf")
    best_state = None
    start = time.time()

    print("\n" + "=" * 72)
    print("  PARAMETRIC KEPLER RESIDUAL SURROGATE")
    print("  Baseline: analytic Kepler anomaly solver + neural residual")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Train e range: 0.0 to {dataset_config.e_max_train:.2f}")
    print(f"  Epochs: {config.epochs:,}")
    print("=" * 72)

    for epoch in range(config.epochs):
        model.train()
        e_cap = config.eccentricity_cap(epoch, dataset_config.e_max_train)
        weights = config.loss_weights(epoch)

        t_data, p_data, y_data = dataset.get_data_batch(e_cap)
        t0, p0, y0 = dataset.get_initial_batch(e_cap)
        pred = model(t_data, p_data)

        l_data = data_loss(pred, y_data)
        l_initial = initial_condition_loss(model, t0, p0, y0)
        zero = torch.zeros((), dtype=torch.float32, device=device)

        if weights["physics"] > 0:
            t_colloc, p_colloc = dataset.get_collocation_batch(e_cap)
            l_physics = physics_loss(model, t_colloc, p_colloc)
        else:
            l_physics = zero

        l_energy = energy_loss(pred) if weights["energy"] > 0 else zero
        l_momentum = momentum_loss(pred, p_data) if weights["momentum"] > 0 else zero

        loss = (
            weights["data"] * l_data
            + weights["initial"] * l_initial
            + weights["physics"] * l_physics
            + weights["energy"] * l_energy
            + weights["momentum"] * l_momentum
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch > config.conservation_ramp_end and loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % config.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            history["epoch"].append(epoch)
            history["loss_total"].append(loss.item())
            history["loss_data"].append(l_data.item())
            history["loss_initial"].append(l_initial.item())
            history["loss_physics"].append(l_physics.item())
            history["loss_energy"].append(l_energy.item())
            history["loss_momentum"].append(l_momentum.item())
            history["eccentricity_cap"].append(e_cap)
            history["lr"].append(lr)
            print(
                f"  Epoch {epoch:5d}/{config.epochs} | "
                f"e_cap={e_cap:.3f} | total={loss.item():.3e} | "
                f"data={l_data.item():.3e} | phys={l_physics.item():.3e} | "
                f"lr={lr:.1e} | {time.time() - start:.0f}s"
            )

        if epoch % config.eval_interval == 0 and epoch > 0:
            metrics = evaluate(model, dataset)
            print(
                f"       Eval in-domain mean: {metrics['in_domain_mean_error_km']:.2f} km | "
                f"OOD mean: {metrics['out_of_domain_mean_error_km']:.2f} km"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, dataset)

    print("\n" + "=" * 72)
    print("  FINAL PARAMETRIC EVALUATION")
    for case in metrics["cases"]:
        print(
            f"  {case['label']:<16} {case['split']:<14} "
            f"e={case['eccentricity']:.2f} | "
            f"mean={case['position_error_mean_km']:.2f} km | "
            f"max={case['position_error_max_km']:.2f} km"
        )
    print("=" * 72)

    if export:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        web_models_dir = os.path.join(project_root, "web", "models")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(web_models_dir, exist_ok=True)
        output_path = os.path.join(models_dir, "parametric_kepler.json")
        export_parametric_model(model, dataset_config, metrics, history, output_path)
        shutil.copy2(output_path, os.path.join(web_models_dir, "parametric_kepler.json"))
        print("  [OK] Mirrored parametric model to web/models/parametric_kepler.json")

    return model, metrics, history


def main():
    parser = argparse.ArgumentParser(description="Train the parametric Kepler surrogate")
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--num-fourier-freq", type=int, default=8)
    parser.add_argument("--train-orbits", type=int, default=72)
    parser.add_argument("--steps-per-orbit", type=int, default=2400)
    parser.add_argument("--e-max", type=float, default=0.80)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    config = ParametricTrainingConfig(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        num_fourier_freq=args.num_fourier_freq,
    )
    ratio = args.epochs / 20_000
    config.physics_ramp_start = int(1200 * ratio)
    config.physics_ramp_end = max(config.physics_ramp_start + 1, int(5000 * ratio))
    config.conservation_ramp_start = int(4000 * ratio)
    config.conservation_ramp_end = max(config.conservation_ramp_start + 1, int(9000 * ratio))
    config.log_interval = max(20, int(100 * ratio))
    config.eval_interval = max(200, int(1000 * ratio))

    dataset_config = ParametricDatasetConfig(
        n_train_orbits=args.train_orbits,
        steps_per_orbit=args.steps_per_orbit,
        e_max_train=args.e_max,
    )
    train(config, dataset_config, args.export)


if __name__ == "__main__":
    main()
