"""
trainer.py - PINN training loop with curriculum learning and loss scheduling.

Training strategy:
  1. CURRICULUM LEARNING: Start training on 1 orbit, then gradually extend
     to 3 and 5 orbits.  The network learns short-term dynamics first,
     then generalizes to longer time horizons.

  2. LOSS WEIGHT SCHEDULING: Begin with high data loss weight to anchor
     the network, then ramp up the physics loss to enforce ODE constraints.
     Conservation losses (energy, angular momentum) come last as a polish.

  3. COSINE ANNEALING LR: Smoothly decays learning rate from 1e-3 to 1e-5
     over the full training run.  No sharp drops or restarts.

  4. BEST MODEL TRACKING: Saves the model state with lowest total loss
     and restores it at the end of training.
"""

import math
import time
import torch
import torch.optim as optim

from model import PINN
from losses import (
    compute_data_loss,
    compute_initial_condition_loss,
    compute_physics_loss,
    compute_energy_loss,
    compute_momentum_loss,
    safe_radius,
)
from dataset import OrbitalDataset
from config import OrbitalScenario, TrainingConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINNTrainer:
    """
    Trains a PINN on a specific orbital scenario.

    Usage:
        trainer = PINNTrainer(scenario, config)
        model, history = trainer.train()
    """

    def __init__(self, scenario: OrbitalScenario, config: TrainingConfig):
        self.scenario = scenario
        self.config = config

        # Data
        print(f"\nGenerating ground-truth trajectory...")
        self.dataset = OrbitalDataset(scenario, config)

        # Model
        self.model = PINN(
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            num_fourier_freq=config.num_fourier_freq,
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr_min
        )

        # History
        self.history = {
            "epoch": [],
            "loss_total": [],
            "loss_data": [],
            "loss_initial": [],
            "loss_physics": [],
            "loss_energy": [],
            "loss_momentum": [],
            "position_error_km": [],
            "energy_drift_pct": [],
            "momentum_drift_pct": [],
            "lr": [],
        }

        self.best_loss = float("inf")
        self.best_state_dict = None
        self.best_epoch = None
        self.best_selection_start = min(
            config.conservation_ramp_end,
            max(0, int(0.8 * config.epochs)),
        )

        self.t_initial = torch.zeros((1, 1), dtype=torch.float32, device=device)
        self.initial_state = torch.tensor(
            scenario.initial_state_normalized,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        self.target_momentum = math.sqrt(max(0.0, 1.0 - scenario.eccentricity**2))

    def train(self):
        """
        Run the full training loop.

        Returns:
            model:   Trained PINN (loaded with best weights)
            history: Dict of training metrics over epochs
        """
        cfg = self.config
        n_params = self.model.count_parameters()

        print(f"\n{'='*65}")
        print(f"  PINN TRAINING -- {self.scenario.name}")
        print(f"{'-'*65}")
        print(f"  Orbit:       {self.scenario.periapsis_alt_km:.0f} x "
              f"{self.scenario.apoapsis_alt_km:.0f} km "
              f"(e = {self.scenario.eccentricity:.4f})")
        print(f"  Period:      {self.scenario.period:.0f} s "
              f"({self.scenario.period/60:.1f} min)")
        print(f"  Network:     {cfg.hidden_layers} x {cfg.hidden_dim} tanh "
              f"+ {cfg.num_fourier_freq} Fourier freq "
              f"= {n_params:,} params")
        print(f"  Training:    {cfg.epochs:,} epochs, "
              f"LR {cfg.learning_rate}->{cfg.lr_min}")
        print(f"  Data:        {cfg.n_data_points} trajectory points, "
              f"{cfg.n_collocation} collocation points")
        print(f"{'='*65}\n")

        start_time = time.time()

        for epoch in range(cfg.epochs):
            self.model.train()

            # Curriculum: determine time domain
            n_orbits = min(
                cfg.get_curriculum_n_orbits(epoch),
                self.scenario.n_orbits_train
            )

            # Get data
            t_data, states_gt = self.dataset.get_training_data(n_orbits)
            t_colloc = self.dataset.get_collocation_points(n_orbits)

            # Loss weights
            weights = cfg.get_loss_weights(epoch)

            # Forward pass
            self.optimizer.zero_grad()

            # Data loss
            pred_data = self.model(t_data)
            loss_data = compute_data_loss(pred_data, states_gt)

            # Initial condition loss. This hard boundary condition prevents
            # phase/offset errors from hiding inside trajectory MSE.
            pred_initial = self.model(self.t_initial)
            loss_initial = compute_initial_condition_loss(
                pred_initial, self.initial_state
            )

            zero = torch.zeros((), dtype=torch.float32, device=device)

            # Physics loss (ODE residuals via autograd). Skip it entirely
            # while its scheduled weight is zero so singular residuals cannot
            # create inf/nan values during strict data pretraining.
            if weights["physics"] > 0.0:
                loss_physics = compute_physics_loss(
                    self.model, t_colloc, self.scenario.x_scale
                )
            else:
                loss_physics = zero

            # Conservation losses
            if weights["energy"] > 0.0:
                loss_energy = compute_energy_loss(pred_data)
            else:
                loss_energy = zero

            if weights["momentum"] > 0.0:
                loss_momentum = compute_momentum_loss(
                    pred_data, self.target_momentum
                )
            else:
                loss_momentum = zero

            # Weighted total loss
            loss_total = (
                weights["data"]     * loss_data
                + weights["initial"] * loss_initial
                + weights["physics"]  * loss_physics
                + weights["energy"]   * loss_energy
                + weights["momentum"] * loss_momentum
            )

            # Backward pass
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Track best model
            current_loss = loss_total.item()
            if epoch >= self.best_selection_start and current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                self.best_state_dict = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            # Logging
            if epoch % cfg.log_interval == 0:
                self._log_progress(
                    epoch, loss_total, loss_data, loss_initial, loss_physics,
                    loss_energy, loss_momentum, n_orbits, start_time
                )

            # Periodic evaluation
            if epoch % cfg.eval_interval == 0:
                self._evaluate(epoch)

        # Restore best model
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        else:
            self.best_loss = current_loss
            self.best_epoch = cfg.epochs - 1

        elapsed = time.time() - start_time
        print(f"\n{'='*65}")
        print(f"  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Best total loss: {self.best_loss:.6e} at epoch {self.best_epoch}")
        print(f"{'='*65}")

        return self.model, self.history

    def _log_progress(self, epoch, loss_total, loss_data, loss_initial,
                      loss_physics, loss_energy, loss_momentum,
                      n_orbits, start_time):
        """Record and print training progress."""
        lr = self.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        self.history["epoch"].append(epoch)
        self.history["loss_total"].append(loss_total.item())
        self.history["loss_data"].append(loss_data.item())
        self.history["loss_initial"].append(loss_initial.item())
        self.history["loss_physics"].append(loss_physics.item())
        self.history["loss_energy"].append(loss_energy.item())
        self.history["loss_momentum"].append(loss_momentum.item())
        self.history["lr"].append(lr)

        # Curriculum stage indicator
        stage = "*" if n_orbits >= self.scenario.n_orbits_train else "o"

        print(
            f"  {stage} Epoch {epoch:5d}/{self.config.epochs} | "
            f"Total {loss_total.item():.2e} | "
            f"Data {loss_data.item():.2e} | "
            f"IC {loss_initial.item():.2e} | "
            f"Phys {loss_physics.item():.2e} | "
            f"E {loss_energy.item():.2e} | "
            f"L {loss_momentum.item():.2e} | "
            f"orb={n_orbits:.0f} | "
            f"lr={lr:.1e} | {elapsed:.0f}s"
        )

    @torch.no_grad()
    def _evaluate(self, epoch):
        """Compute and log evaluation metrics."""
        self.model.eval()

        t_eval, states_gt = self.dataset.get_eval_data()
        pred = self.model(t_eval)

        # Position error [km]
        pred_phys = self.dataset.denormalize_state(pred)
        gt_phys   = self.dataset.denormalize_state(states_gt)

        pos_error = torch.sqrt(
            (pred_phys[:, 0] - gt_phys[:, 0])**2
            + (pred_phys[:, 1] - gt_phys[:, 1])**2
        ) / 1e3   # -> km

        mean_pos_err = pos_error.mean().item()
        max_pos_err  = pos_error.max().item()

        # Energy conservation
        x, y, vx, vy = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        r = safe_radius(x, y)
        energy = 0.5 * (vx**2 + vy**2) - 1.0 / r
        max_energy_drift = (
            torch.max(torch.abs(energy - energy[0])) / torch.abs(energy[0])
        ).item() * 100

        # Angular momentum conservation
        ang_mom = x * vy - y * vx
        max_mom_drift = (
            torch.max(torch.abs(ang_mom - ang_mom[0])) / torch.abs(ang_mom[0])
        ).item() * 100

        # Store
        self.history["position_error_km"].append(mean_pos_err)
        self.history["energy_drift_pct"].append(max_energy_drift)
        self.history["momentum_drift_pct"].append(max_mom_drift)

        # Print every 4th eval (less verbose)
        if epoch % (self.config.eval_interval * 2) == 0:
            print(
                f"       >> EVAL | "
                f"Pos err: {mean_pos_err:.3f} km (max {max_pos_err:.2f}) | "
                f"Energy drift: {max_energy_drift:.4f}% | "
                f"Mom drift: {max_mom_drift:.4f}%"
            )

        self.model.train()
