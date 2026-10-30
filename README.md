# KeplerLab: PINN Failure Analysis for Orbital Dynamics

KeplerLab is a Physics-Informed Neural Network project for planar spacecraft dynamics. Its browser demo explores model training, trajectory predictions, and the limits of numerical comparisons.

**[Open the historical demo](https://kepler-lab.vercel.app)** · **[Read the later numerical audit](https://github.com/dlowzzxx/NOETHER/blob/main/PINN-REVIVAL.md#2-planning-audit--legacy-headline-quarantined)**

The goal is not just a pretty orbit. The goal is to make the debugging work legible.

## What The Demo Shows

- A single orbit board with a Verlet reference, PINN prediction, error vectors, periapsis/apoapsis labels, and optional collocation overlays.
- A PINN Autopsy stepper: Gradient Shock, Strict Pretraining, Safe Radius, Periapsis Sampling, Final Model.
- A diagnostics strip for position error, energy drift, angular momentum drift, inference speed, and status.
- A Model Card tab describing architecture, loss schedule, normalization, metrics, and limitations.
- Honest evidence labels: final traces are measured from exported model weights; earlier autopsy stages are diagnostic reconstructions until dedicated ablation weights are trained.

## Evidence update

The original demo's summary figures are historical, not verified error bounds. Its builder copied fixed-model summary values from model metadata rather than recomputing them from the displayed trace, and its finite-step Verlet reference accumulates phase error on high-eccentricity cases.

A later [NOETHER audit](https://github.com/dlowzzxx/NOETHER/blob/main/PINN-REVIVAL.md#2-planning-audit--legacy-headline-quarantined) re-evaluated the same exported high-e weights against analytic Kepler motion. Its [regenerated artifact](https://github.com/dlowzzxx/NOETHER/blob/main/app/public/data/pinn/evidence/fixed-high-e-analytic.v1.json) reports 7.08 km mean and 18.16 km maximum position error over 1,441 samples in one synthetic, fitted orbit. This is not evidence of general orbit prediction or a flight-ready model.

The legacy `e = 0.87` out-of-domain case is quarantined: its nominal periapsis lies about 3,197 km inside Earth, and the coarse reference is about 1,464 km from analytic Kepler at one orbit. Its old model-versus-reference error cannot establish a model failure. The raw historical traces remain available for inspection.

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

Each trace includes downsampled time, PINN/reconstruction state, Verlet state, position error, energy drift, momentum drift, residual values, and collocation samples. Final traces use exported model JSON in `models/`, but the original summary values and reference have the limitations described above.

Regenerate evidence after exporting models:

```bash
node scripts/build_evidence.js
node scripts/build_surrogate_evidence.js
node scripts/check_evidence.js
```

## Hybrid Surrogate Solver

The `Surrogate Solver` tab is a second path: a parameter-conditioned hybrid model that predicts a family of Keplerian orbits instead of one fixed scenario. It is separate from the fixed-trajectory Failure Lab.

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

The legacy browser port omits the final `tanh` in the condition encoder, so its outputs do not exactly match the exported training architecture. Its original trace-replay figures use the coarse reference and should not be taken as validation. The later audit uses corrected inference and an analytic reference; see its [in-domain](https://github.com/dlowzzxx/NOETHER/blob/main/app/public/data/pinn/evidence/hybrid-in-domain-analytic.v1.json) and [physical out-of-domain](https://github.com/dlowzzxx/NOETHER/blob/main/app/public/data/pinn/evidence/hybrid-ood-e085-a45m-analytic.v1.json) artifacts for the stated scenarios and limits.

## Project Structure

```text
PINN/
  training/         PyTorch PINN, numerical reference, losses, export
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
