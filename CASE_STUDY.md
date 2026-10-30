# KeplerLab Case Study

## One-Line Summary

KeplerLab is an orbital-dynamics PINN project that explores model failures, physics-aware fixes, and browser inference. A later numerical audit exposed important limits in the original evaluation.

## Why This Exists

The goal was not to make a decorative orbit animation. The goal was to answer a harder engineering question:

Can a small neural network learn spacecraft trajectories while obeying Newtonian dynamics closely enough to be useful as a fast browser-side surrogate?

## Failure Mode

One investigated failure mode is applying the physics loss before the network has learned the orbit geometry. At random initialization, predicted position can land near `r = 0`. The two-body residual contains `x / r^3`, so an early physics update can explode and push tanh units into saturation.

KeplerLab illustrates this as the `Gradient Shock` stage. That stage is a diagnostic reconstruction, not a separately trained and measured ablation.

## Fixes

- Strict pretraining: train on data and initial conditions first, then ramp physics.
- Safe radius: use `sqrt(x^2 + y^2 + 1e-6)` inside the residual.
- Exact invariants: penalize drift from normalized Keplerian energy `-0.5` and angular momentum `sqrt(1 - e^2)`.
- Periapsis sampling: put half the collocation points near periapsis so high-e dynamics are not averaged away.
- Evidence export: generate compact traces from exported JSON weights and verify them with a separate Node checker.

## Evidence update

The original fixed-model summary metrics were copied from metadata, not recomputed from the displayed samples, and the finite-step reference accumulates phase error on high-eccentricity cases. A later [NOETHER audit](https://github.com/dlowzzxx/NOETHER/blob/main/PINN-REVIVAL.md#2-planning-audit--legacy-headline-quarantined) re-evaluated the same high-e weights against analytic Kepler motion: 7.08 km mean and 18.16 km maximum position error across 1,441 samples in one fitted, synthetic orbit. The [generated artifact](https://github.com/dlowzzxx/NOETHER/blob/main/app/public/data/pinn/evidence/fixed-high-e-analytic.v1.json) records the method and limitations.

## What Is Measured

The final step in the UI uses exported model weights in `models/*.json` and compares them with a numerical Velocity Verlet reference. Earlier autopsy stages are diagnostic reconstructions, not separately trained models. The historical UI summaries are not validated accuracy claims.

## Generalization Path

The parametric surrogate is separate from the fixed-trajectory Failure Lab. It uses a hybrid residual architecture: solve Kepler's equation for eccentric anomaly first, compute the normalized two-body state, then let a small neural network learn bounded corrections.

The original browser port omits the final `tanh` in the condition encoder, so it does not exactly match the exported training architecture. The legacy `e = 0.87` case crosses Earth and its coarse reference accumulates substantial phase error; the old result is quarantined rather than used as a model-failure verdict. The later audit documents corrected inference and an analytic reference.

## Reproduce

```bash
python training/train.py --all --epochs 20000 --export
node scripts/build_evidence.js
node scripts/build_surrogate_evidence.js
node scripts/check_evidence.js
node serve.js
```

Open `http://localhost:8000/web/`.
