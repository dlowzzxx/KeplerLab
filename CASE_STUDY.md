# KeplerLab Case Study

## One-Line Summary

KeplerLab is a from-scratch orbital-dynamics PINN project that turns a failed neural physics model into a measured failure-analysis lab: diagnose the failure, apply physics-aware fixes, export browser weights, and prove the result against Velocity Verlet truth.

## Why This Exists

The goal was not to make a decorative orbit animation. The goal was to answer a harder engineering question:

Can a small neural network learn spacecraft trajectories while obeying Newtonian dynamics closely enough to be useful as a fast browser-side surrogate?

## Failure Mode

The first model failed because the physics loss was allowed to act before the network had learned the orbit geometry. At random initialization, predicted position can land near `r = 0`. The two-body residual contains `x / r^3`, so the first physics update can explode and push tanh units into saturation.

KeplerLab exposes this as the `Gradient Shock` stage instead of hiding it.

## Fixes

- Strict pretraining: train on data and initial conditions first, then ramp physics.
- Safe radius: use `sqrt(x^2 + y^2 + 1e-6)` inside the residual.
- Exact invariants: penalize drift from normalized Keplerian energy `-0.5` and angular momentum `sqrt(1 - e^2)`.
- Periapsis sampling: put half the collocation points near periapsis so high-e dynamics are not averaged away.
- Evidence export: generate compact traces from exported JSON weights and verify them with a separate Node checker.

## Measured Results

| Scenario | Mean position error | Max position error | Energy drift max | Momentum drift max | Browser speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circular LEO | 0.332 km | 1.005 km | 0.0356% | 0.0178% | 733x |
| Artemis-style Elliptical | 0.371 km | 1.149 km | 0.0430% | 0.0193% | 790x |
| GTO High-E | 5.927 km | 42.952 km | 0.6808% | 0.1206% | 365x |

The high-e result is the important stress test: the model goes from a 23,000+ km gradient-shock reconstruction to 5.927 km mean error after the training fixes.

## What Is Measured

The final step in the UI is generated from exported model weights in `models/*.json`, then compared against a Velocity Verlet trajectory generated independently in the evidence builder. Earlier autopsy stages are labeled diagnostic reconstructions, so the demo does not pretend every failure step came from a separately trained model unless those ablation weights are added later.

## Generalization Path

The parametric surrogate is intentionally separated from the validated Failure Lab. The first version conditioned on eccentricity and semi-major axis with Fourier time features only, and it failed on high-e motion. The fix is a hybrid residual architecture: solve Kepler's equation for eccentric anomaly first, compute the exact normalized two-body state, then let a small neural network learn bounded corrections.

That distinction matters. The project does not hide the fact that a pure Fourier surrogate struggled; it turns the miss into an engineering decision about coordinates, conditioning, and where a neural model actually belongs. The current exported surrogate is usable inside its trained eccentricity range: the high-e trace replay measures 9.10 km mean error and 39.71 km max error. An extreme `e = 0.87` case remains out-of-domain and is labeled that way in the UI.

## Reproduce

```bash
python training/train.py --all --epochs 20000 --export
node scripts/build_evidence.js
node scripts/build_surrogate_evidence.js
node scripts/check_evidence.js
node serve.js
```

Open `http://localhost:8000/web/`.
