"""
train.py -- Main entry point for PINN training.

Usage:
    # Train all scenarios with default settings:
    python train.py --all --export

    # Train a single scenario:
    python train.py --scenario circular_leo --epochs 15000 --export

    # Quick test run:
    python train.py --scenario circular_leo --epochs 500 --no-plots

Orchestrates: data generation -> training -> evaluation -> export.
"""

import argparse
import os
import shutil
import sys
import time

# Force UTF-8 output on Windows to handle special characters in logs
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import SCENARIOS, TrainingConfig
from trainer import PINNTrainer
from evaluate import compute_final_metrics, generate_plots
from export import export_model


def train_scenario(scenario_name: str, config: TrainingConfig,
                   models_dir: str, results_dir: str,
                   web_models_dir: str,
                   do_export: bool = True, do_plots: bool = True):
    """Train PINN for a single orbital scenario."""
    scenario = SCENARIOS[scenario_name]

    # -- Train -------------------------------------------------------
    trainer = PINNTrainer(scenario, config)
    model, history = trainer.train()

    # -- Evaluate ----------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  FINAL EVALUATION -- {scenario.name.upper().replace('_', ' ')}")
    print(f"{'='*65}")

    metrics = compute_final_metrics(model, trainer.dataset, scenario)

    print(f"\n  Position error:  {metrics['position_error_mean_km']:.4f} km "
          f"(max {metrics['position_error_max_km']:.4f} km)")
    print(f"  Velocity error:  {metrics['velocity_error_mean_ms']:.4f} m/s "
          f"(max {metrics['velocity_error_max_ms']:.4f} m/s)")
    print(f"  Energy drift:    {metrics['energy_drift_max_pct']:.6f}%")
    print(f"  Momentum drift:  {metrics['momentum_drift_max_pct']:.6f}%")
    print(f"  -------------------------------------------------")
    print(f"  PINN inference:  {metrics['pinn_inference_ms']:.3f} ms")
    print(f"  Verlet integr.:  {metrics['verlet_integration_ms']:.3f} ms")
    print(f"  Speedup:         {metrics['speedup_factor']:.1f}x")

    # -- Generate plots ----------------------------------------------
    if do_plots:
        print(f"\n  Generating plots...")
        generate_plots(model, trainer.dataset, scenario, history, results_dir)

    # -- Export model ------------------------------------------------
    if do_export:
        print(f"\n  Exporting model...")
        output_path = os.path.join(models_dir, f"{scenario.name}.json")
        export_model(model, scenario, metrics, history, output_path)
        os.makedirs(web_models_dir, exist_ok=True)
        web_output_path = os.path.join(web_models_dir, f"{scenario.name}.json")
        shutil.copy2(output_path, web_output_path)
        print(f"  [OK] Mirrored web model to {web_output_path}")

    return model, metrics, history


def main():
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed Neural Network for orbital dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --scenario circular_leo --export
  python train.py --all --export
  python train.py --scenario elliptical --epochs 25000 --export
        """
    )
    parser.add_argument(
        "--scenario", type=str, default="circular_leo",
        choices=list(SCENARIOS.keys()),
        help="Orbital scenario to train on (default: circular_leo)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train all scenarios sequentially"
    )
    parser.add_argument(
        "--epochs", type=int, default=15000,
        help="Number of training epochs (default: 15000)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="Hidden layer width (default: 64)"
    )
    parser.add_argument(
        "--hidden-layers", type=int, default=4,
        help="Number of hidden layers (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export trained model weights as JSON"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating evaluation plots"
    )

    args = parser.parse_args()

    # -- Build config ------------------------------------------------
    config = TrainingConfig(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        learning_rate=args.lr,
    )

    # Default 15k schedule: 1 orbit -> 3 orbits at epoch 3000,
    # then 5 orbits at epoch 8000. Scale that shape for custom runs.
    ratio = args.epochs / 15_000
    config.curriculum_stages = [
        {"epoch_start": 0,                   "n_orbits": 1.0},
        {"epoch_start": int(3_000 * ratio),  "n_orbits": 3.0},
        {"epoch_start": int(8_000 * ratio),  "n_orbits": 5.0},
    ]
    config.physics_ramp_start = int(1_000 * ratio)
    config.physics_ramp_end = max(config.physics_ramp_start + 1, int(4_000 * ratio))
    config.conservation_ramp_start = int(3_000 * ratio)
    config.conservation_ramp_end = max(
        config.conservation_ramp_start + 1,
        int(7_000 * ratio),
    )
    config.log_interval = max(10, int(50 * ratio))
    config.eval_interval = max(50, int(500 * ratio))

    # -- Output directories ------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir   = os.path.join(project_root, "models")
    results_dir  = os.path.join(project_root, "results")
    web_models_dir = os.path.join(project_root, "web", "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(web_models_dir, exist_ok=True)

    # -- Banner ------------------------------------------------------
    print("\n" + "="*65)
    print("  +=====================================================+")
    print("  |  PHYSICS-INFORMED NEURAL NETWORK                    |")
    print("  |  Spacecraft Orbital Dynamics                        |")
    print("  |  ------------------------------------------------- |")
    print("  |  Newton's Law of Gravitation x Deep Learning        |")
    print("  +=====================================================+")
    print("="*65)

    # -- Train -------------------------------------------------------
    total_start = time.time()

    if args.all:
        scenarios = list(SCENARIOS.keys())
        print(f"\n  Training all {len(scenarios)} scenarios: {', '.join(scenarios)}")
        all_metrics = {}
        for name in scenarios:
            _, metrics, _ = train_scenario(
                name, config, models_dir, results_dir, web_models_dir,
                do_export=args.export, do_plots=not args.no_plots
            )
            all_metrics[name] = metrics

        # Print summary table
        print(f"\n\n{'='*65}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Scenario':<20} {'Pos err (km)':<14} {'E drift %':<12} {'Speedup':<10}")
        print(f"  {'-'*20} {'-'*14} {'-'*12} {'-'*10}")
        for name, m in all_metrics.items():
            print(f"  {name:<20} {m['position_error_mean_km']:<14.4f} "
                  f"{m['energy_drift_max_pct']:<12.6f} {m['speedup_factor']:<10.1f}x")
    else:
        train_scenario(
            args.scenario, config, models_dir, results_dir, web_models_dir,
            do_export=args.export, do_plots=not args.no_plots
        )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Done!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
