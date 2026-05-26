# KeplerLab: PINN Failure Analysis for Orbital Dynamics

KeplerLab is a from-scratch Physics-Informed Neural Network project for planar spacecraft dynamics. The live demo is framed as an engineering failure lab: it shows the model fail, diagnoses why, applies fixes, and proves the final exported model against Velocity Verlet truth with trajectory, invariant, residual, and speed evidence.

The goal is not just a pretty orbit. The goal is to make the debugging work legible.

## What The Demo Shows

- A single orbit board with Verlet truth, PINN prediction, error vectors, periapsis/apoapsis labels, and optional collocation overlays.
- A PINN Autopsy stepper: Gradient Shock, Strict Pretraining, Safe Radius, Periapsis Sampling, Final Model.
- A diagnostics strip for position error, energy drift, angular momentum drift, inference speed, and status.
- A Model Card tab describing architecture, loss schedule, normalization, metrics, and limitations.
- Honest evidence labels: final traces are measured from exported model weights; earlier autopsy stages are diagnostic reconstructions until dedicated ablation weights are trained.

## Current Measured Results

| Scenario | Mean position error | Max position error | Energy drift max | Momentum drift max | Browser speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circular LEO | 0.332 km | 1.005 km | 0.0356% | 0.0178% | 733x |
| Artemis-style Elliptical | 0.371 km | 1.149 km | 0.0430% | 0.0193% | 790x |
| GTO High-E | 5.927 km | 42.952 km | 0.6808% | 0.1206% | 365x |

The high-e orbit is the stress test. After periapsis-focused collocation and a 20,000-epoch run, the final exported model is below 6 km mean position error while the gradient-shock reconstruction is over 23,000 km.

## Physics

For a spacecraft orbiting Earth in the planar two-body problem:

```text
dx/dt  = vx
dy/dt  = vy
dvx/dt = -mu * x / (x^2 + y^2)^(3/2)
dvy/dt = -mu * y / (x^2 + y^2)^(3/2)
```

where `mu = G * M_earth`.

Each scenario is non-dimensionalized:

```text
length scale   = semi-major axis a
velocity scale = sqrt(mu / a)
time scale     = sqrt(a^3 / mu)
```

In normalized coordinates, `mu = 1` and one orbit has period `2*pi`. The specific orbital energy target is exactly `-0.5`, and angular momentum is `sqrt(1 - e^2)`.

## Model

```text
t_norm -> Fourier features -> MLP (4 x 64 tanh) -> [x, y, vx, vy]
```

- Fourier features use periodic harmonics so the MLP can represent orbital motion cleanly.
- Tanh activations keep the autograd derivatives smooth for the ODE residual.
- The default model has 14,084 trainable parameters.
- Exported JSON weights remain compatible with browser-side inference in vanilla JavaScript.

## Failure Fixes

The first failure was an initialization singularity: random network outputs could land near `r = 0`, making `x / r^3` explode before the model learned the orbit geometry.

The fixed schedule is:

```text
epochs 0-999:     data + initial-condition loss only
epochs 1000-4000: physics loss ramps 0 -> 1
epochs 3000-7000: energy and momentum losses ramp 0 -> 0.1
epochs 7000+:     full objective
```

The loss also uses a safe radius:

```python
r = torch.sqrt(x**2 + y**2 + 1e-6)
```

For high-eccentricity orbits, collocation sampling is mixed: about half uniform across the orbit and half concentrated near periapsis, where Kepler's second law makes the dynamics hardest.

## Evidence Interface

The web demo reads compact evidence files:

```text
web/data/experiments.json
web/data/surrogate.json
web/data/traces/{scenario}_{experiment}.json
```

Each trace includes downsampled time, PINN/reconstruction state, Verlet state, position error, energy drift, momentum drift, residual values, and collocation samples. Final traces are generated from the exported model JSON in `models/`.

Regenerate evidence after exporting models:

```bash
node scripts/build_evidence.js
node scripts/build_surrogate_evidence.js
node scripts/check_evidence.js
```

## Hybrid Surrogate Solver

The `Surrogate Solver` tab is a second path: a parameter-conditioned hybrid model that predicts a continuous family of Keplerian orbits instead of one fixed scenario. It is intentionally separate from the validated Failure Lab.

The first parametric model was a useful failure: a Fourier-only network underfit high-eccentricity motion because periapsis is too sharp in mean-anomaly time. The current v2 model uses the physically natural coordinate:

```text
mean anomaly M -> eccentric anomaly E, where M = E - e sin(E)
exact Kepler state(E, e) + neural residual(t, e, a) -> [x, y, vx, vy]
```

That makes this path an honest hybrid physics-neural surrogate, not a black-box replacement for astrodynamics.

Train and export it with:

```bash
python training/train_parametric.py --epochs 5000 --export
node scripts/build_surrogate_evidence.js
```

The exported model contract is:

```text
Model(t_norm, eccentricity, semi_major_axis_feature)
    -> Kepler baseline + bounded neural residual
    -> [x, y, vx, vy]
```

Current trace-replay validation for the exported surrogate:

| Surrogate case | Domain | Mean position error | Max position error |
| --- | --- | ---: | ---: |
| Seen low-e | in-domain | 3.26 km | 7.12 km |
| Seen mid-e | in-domain | 5.47 km | 16.77 km |
| Seen high-e | in-domain | 9.10 km | 39.71 km |
| OOD extreme-e | out-of-domain | 173.01 km | 1462.74 km |

The surrogate is now usable inside the trained eccentricity range, including the high-e validation case. The extreme-e case is deliberately labeled out-of-domain in the UI. The recruiter-facing headline remains the per-scenario Failure Lab model set, because those exported PINNs are the highest-accuracy measured results.

## Project Structure

```text
PINN/
  training/         PyTorch PINN, Velocity Verlet truth, losses, export
  models/           exported JSON weights
  web/              vanilla HTML/CSS/Canvas/JS demo
  web/data/         compact evidence manifest and traces
  scripts/          evidence builder and checks
  results/          generated training and trajectory plots
  serve.js          no-cache static server
```

## Reproduce

```bash
cd training
pip install -r requirements.txt
python train.py --all --epochs 20000 --export
cd ..
node scripts/build_evidence.js
node scripts/build_surrogate_evidence.js
node scripts/check_evidence.js
node serve.js
```

Open `http://localhost:8000/web/`.

Do not open `web/index.html` directly as a `file://` tab. Browser security blocks `fetch()` from loading the JSON evidence, and the app will show a server-required notice.

## References

- Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", 2019.
- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions", 2020.
- Standard two-body orbital mechanics, Keplerian elements, and the vis-viva equation.
