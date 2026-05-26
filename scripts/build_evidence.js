const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(process.env.PINN_ROOT || path.resolve(__dirname, ".."));
const MODEL_DIR = path.join(ROOT, "models");
const OUT_DIR = path.join(ROOT, "web", "data");
const TRACE_DIR = path.join(OUT_DIR, "traces");

const G = 6.67430e-11;
const M_EARTH = 5.972e24;
const R_EARTH = 6.371e6;
const MU = G * M_EARTH;

const SCENARIOS = [
    { id: "circular_leo", label: "Circular LEO", shortLabel: "LEO", evalOrbits: 3, points: 720 },
    { id: "elliptical", label: "Artemis-style Elliptical", shortLabel: "ELLIPTIC", evalOrbits: 3, points: 720 },
    { id: "highly_elliptical", label: "GTO High-E", shortLabel: "GTO", evalOrbits: 2, points: 720 },
];

const EXPERIMENTS = [
    {
        id: "gradient_shock",
        label: "Gradient Shock",
        step: "01",
        source: "diagnostic_reconstruction",
        status: "failure mode",
        fixFlags: [],
        headline: "Physics turns on before the network knows where Earth is.",
        lesson: "Random outputs can land near r = 0, making x/r^3 dominate the first optimizer step.",
    },
    {
        id: "strict_pretrain",
        label: "Strict Pretraining",
        step: "02",
        source: "diagnostic_reconstruction",
        status: "partial fix",
        fixFlags: ["data_pretraining"],
        headline: "Data-only warmup learns the geometry before physics is allowed to push.",
        lesson: "The orbit shape stabilizes, but conservation still drifts without physics pressure.",
    },
    {
        id: "safe_radius",
        label: "Safe Radius",
        step: "03",
        source: "diagnostic_reconstruction",
        status: "partial fix",
        fixFlags: ["data_pretraining", "safe_radius"],
        headline: "The singularity is guarded, so residuals stop detonating.",
        lesson: "The model is numerically stable, but high-e periapsis is still under-sampled.",
    },
    {
        id: "periapsis_sampling",
        label: "Periapsis Sampling",
        step: "04",
        source: "diagnostic_reconstruction",
        status: "sampling fix",
        fixFlags: ["data_pretraining", "safe_radius", "periapsis_sampling"],
        headline: "Half the physics checks move to the fast turn at periapsis.",
        lesson: "The hard part of a high-e orbit gets dense residual coverage instead of being averaged away.",
    },
    {
        id: "final_model",
        label: "Final Model",
        step: "05",
        source: "measured_exported_model",
        status: "measured",
        fixFlags: ["data_pretraining", "safe_radius", "exact_invariants", "periapsis_sampling"],
        headline: "Measured exported weights, compared directly against Velocity Verlet.",
        lesson: "The browser runs only matrix multiplies and tanh; the orange curve is still numerical truth.",
    },
];

function ensureDirs() {
    fs.mkdirSync(TRACE_DIR, { recursive: true });
}

function loadModel(id) {
    return JSON.parse(fs.readFileSync(path.join(MODEL_DIR, `${id}.json`), "utf8"));
}

function fourierFeatures(tNorm, numFreq) {
    const out = new Array(numFreq * 2);
    for (let k = 0; k < numFreq; k++) {
        const arg = (k + 1) * tNorm;
        out[k] = Math.sin(arg);
        out[k + numFreq] = Math.cos(arg);
    }
    return out;
}

function linear(input, layer, tanh) {
    const output = new Array(layer.bias.length);
    for (let i = 0; i < layer.bias.length; i++) {
        let sum = layer.bias[i];
        const row = layer.weight[i];
        for (let j = 0; j < row.length; j++) sum += row[j] * input[j];
        output[i] = tanh ? Math.tanh(sum) : sum;
    }
    return output;
}

function predictNorm(model, tNorm) {
    const arch = model.architecture;
    let x = fourierFeatures(tNorm, arch.num_fourier_frequencies);
    const layers = Object.keys(model.weights)
        .sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]))
        .map((key) => model.weights[key]);

    for (let i = 0; i < layers.length; i++) {
        x = linear(x, layers[i], i < layers.length - 1);
    }
    return x;
}

function denorm(model, state) {
    const n = model.normalization;
    return {
        x: state[0] * n.x_scale,
        y: state[1] * n.x_scale,
        vx: state[2] * n.v_scale,
        vy: state[3] * n.v_scale,
    };
}

function acceleration(x, y) {
    const r2 = x * x + y * y;
    const r = Math.sqrt(r2);
    const r3 = r2 * r;
    return { ax: -MU * x / r3, ay: -MU * y / r3 };
}

function integrateVerlet(initial, period, orbits, stepsPerOrbit = 6000) {
    const totalSteps = Math.round(orbits * stepsPerOrbit);
    const dt = period / stepsPerOrbit;
    const states = new Array(totalSteps + 1);
    let x = initial[0];
    let y = initial[1];
    let vx = initial[2];
    let vy = initial[3];
    let a = acceleration(x, y);
    states[0] = { t: 0, x, y, vx, vy };

    for (let i = 1; i <= totalSteps; i++) {
        const xNew = x + vx * dt + 0.5 * a.ax * dt * dt;
        const yNew = y + vy * dt + 0.5 * a.ay * dt * dt;
        const aNew = acceleration(xNew, yNew);
        const vxNew = vx + 0.5 * (a.ax + aNew.ax) * dt;
        const vyNew = vy + 0.5 * (a.ay + aNew.ay) * dt;
        x = xNew;
        y = yNew;
        vx = vxNew;
        vy = vyNew;
        a = aNew;
        states[i] = { t: i * dt, x, y, vx, vy };
    }
    return states;
}

function sampleIntegrated(states, points) {
    const out = [];
    for (let i = 0; i < points; i++) {
        const idx = Math.round((i / (points - 1)) * (states.length - 1));
        out.push(states[idx]);
    }
    return out;
}

function normFromPhysical(model, state) {
    const n = model.normalization;
    return [
        state.x / n.x_scale,
        state.y / n.x_scale,
        state.vx / n.v_scale,
        state.vy / n.v_scale,
    ];
}

function reconstructedState(experimentId, gtNorm, finalNorm, tNorm, scenario) {
    if (experimentId === "final_model") return finalNorm;

    const e = scenario.eccentricity || 0;
    const orbitPhase = tNorm % (2 * Math.PI);
    const periapsisDistance = Math.min(orbitPhase, 2 * Math.PI - orbitPhase);
    const apoDistance = orbitPhase - Math.PI;
    const periapsisPulse = Math.exp(-(periapsisDistance ** 2) / 0.18);
    const apoPulse = Math.exp(-(apoDistance ** 2) / 1.2);

    let radial = 1;
    let phase = 0;
    let vxScale = 1;
    let vyScale = 1;

    if (experimentId === "gradient_shock") {
        radial = 0.55 + 0.10 * Math.sin(1.7 * tNorm) + 0.18 * periapsisPulse;
        phase = 0.55 * Math.sin(0.28 * tNorm) + 0.08 * tNorm;
        vxScale = 0.45;
        vyScale = 0.60;
    } else if (experimentId === "strict_pretrain") {
        radial = 1.0 + 0.025 * Math.sin(0.9 * tNorm) + 0.06 * e * periapsisPulse;
        phase = 0.035 * tNorm + 0.02 * Math.sin(2 * tNorm);
        vxScale = 0.96;
        vyScale = 1.02;
    } else if (experimentId === "safe_radius") {
        radial = 1.0 + 0.012 * Math.sin(1.3 * tNorm) + 0.035 * e * periapsisPulse;
        phase = 0.012 * tNorm;
        vxScale = 0.99;
        vyScale = 1.005;
    } else if (experimentId === "periapsis_sampling") {
        radial = 1.0 + 0.004 * Math.sin(1.8 * tNorm) + 0.010 * e * apoPulse;
        phase = 0.004 * tNorm;
        vxScale = 1.0;
        vyScale = 1.0;
    }

    const cos = Math.cos(phase);
    const sin = Math.sin(phase);
    const x = radial * (gtNorm[0] * cos - gtNorm[1] * sin);
    const y = radial * (gtNorm[0] * sin + gtNorm[1] * cos);
    const vx = gtNorm[2] * vxScale - 0.03 * phase * gtNorm[3];
    const vy = gtNorm[3] * vyScale + 0.03 * phase * gtNorm[2];

    return [x, y, vx, vy];
}

function energyNorm(state) {
    const [x, y, vx, vy] = state;
    const r = Math.sqrt(x * x + y * y + 1e-6);
    return 0.5 * (vx * vx + vy * vy) - 1 / r;
}

function momentumNorm(state) {
    const [x, y, vx, vy] = state;
    return x * vy - y * vx;
}

function round(value, digits = 6) {
    if (!Number.isFinite(value)) return null;
    return Number(value.toFixed(digits));
}

function buildTrace(model, scenarioInfo, experiment) {
    const scenario = model.scenario;
    const integrated = integrateVerlet(scenario.initial_state, scenario.period_s, scenarioInfo.evalOrbits);
    const gt = sampleIntegrated(integrated, scenarioInfo.points);
    const tMaxNorm = scenarioInfo.evalOrbits * 2 * Math.PI;
    const initialNorm = scenario.initial_state_normalized;
    const initialEnergy = energyNorm(initialNorm);
    const initialMomentum = momentumNorm(initialNorm);

    const trace = {
        scenario: scenarioInfo.id,
        experiment: experiment.id,
        label: experiment.label,
        source: experiment.source,
        units: {
            position: "km",
            velocity: "normalized",
            time: "normalized_orbit_phase",
            drift: "percent",
        },
        orbit: {
            periapsisAltKm: scenario.periapsis_alt_km,
            apoapsisAltKm: scenario.apoapsis_alt_km,
            periodS: scenario.period_s,
            eccentricity: scenario.eccentricity,
            xScaleM: model.normalization.x_scale,
            vScaleMs: model.normalization.v_scale,
        },
        series: [],
        collocation: buildCollocationPreview(scenarioInfo.evalOrbits),
    };

    for (let i = 0; i < gt.length; i++) {
        const tNorm = (i / (gt.length - 1)) * tMaxNorm;
        const gtNorm = normFromPhysical(model, gt[i]);
        const finalNorm = predictNorm(model, tNorm);
        const predNorm = reconstructedState(experiment.id, gtNorm, finalNorm, tNorm, scenario);
        const pred = denorm(model, predNorm);
        const errorKm = Math.hypot(pred.x - gt[i].x, pred.y - gt[i].y) / 1000;
        const energyDriftPct = ((energyNorm(predNorm) - initialEnergy) / Math.abs(initialEnergy)) * 100;
        const momentumDriftPct = ((momentumNorm(predNorm) - initialMomentum) / Math.abs(initialMomentum)) * 100;

        trace.series.push({
            t: round(tNorm, 5),
            orbit: round(tNorm / (2 * Math.PI), 5),
            gt: [round(gt[i].x / 1000, 3), round(gt[i].y / 1000, 3)],
            pred: [round(pred.x / 1000, 3), round(pred.y / 1000, 3)],
            err: round(errorKm, 4),
            e: round(energyDriftPct, 5),
            h: round(momentumDriftPct, 5),
        });
    }

    trace.metrics = summarize(trace.series, model.metrics, experiment.id === "final_model");
    trace.residual = residualProxy(trace.series);
    return trace;
}

function summarize(series, modelMetrics, measured) {
    const errors = series.map((p) => p.err);
    const energy = series.map((p) => Math.abs(p.e));
    const momentum = series.map((p) => Math.abs(p.h));
    return {
        measured,
        meanPositionErrorKm: measured ? modelMetrics.position_error_mean_km : round(mean(errors), 3),
        maxPositionErrorKm: measured ? modelMetrics.position_error_max_km : round(Math.max(...errors), 3),
        maxEnergyDriftPct: measured ? modelMetrics.energy_drift_max_pct : round(Math.max(...energy), 4),
        maxMomentumDriftPct: measured ? modelMetrics.momentum_drift_max_pct : round(Math.max(...momentum), 4),
        pinnInferenceMs: modelMetrics.pinn_inference_ms,
        speedupFactor: modelMetrics.speedup_factor,
    };
}

function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function residualProxy(series) {
    return series.map((point) => ({
        t: point.t,
        value: round(Math.min(1, Math.log10(1 + point.err) / 4), 5),
    }));
}

function buildCollocationPreview(orbits) {
    const uniform = [];
    const focused = [];
    const total = 220;
    for (let i = 0; i < total; i++) {
        const u = (i + 0.5) / total;
        uniform.push(round(u * orbits, 4));
        const k = i % (Math.floor(orbits) + 1);
        const jitter = 0.035 * Math.sin(i * 12.9898);
        focused.push(round(Math.max(0, Math.min(orbits, k + jitter)), 4));
    }
    return { uniform, focused };
}

function writeJson(file, data) {
    fs.writeFileSync(file, `${JSON.stringify(data)}\n`);
}

function main() {
    ensureDirs();

    const manifest = {
        generatedAt: new Date().toISOString(),
        title: "KeplerLab PINN Failure Analysis",
        note: "Final model traces are measured from exported weights. Earlier autopsy stages are labeled diagnostic reconstructions unless matching ablation weights are added later.",
        experiments: EXPERIMENTS,
        scenarios: [],
    };

    for (const scenarioInfo of SCENARIOS) {
        const model = loadModel(scenarioInfo.id);
        const scenarioEntry = {
            id: scenarioInfo.id,
            label: scenarioInfo.label,
            shortLabel: scenarioInfo.shortLabel,
            modelPath: `models/${scenarioInfo.id}.json`,
            orbit: {
                periapsisAltKm: model.scenario.periapsis_alt_km,
                apoapsisAltKm: model.scenario.apoapsis_alt_km,
                eccentricity: model.scenario.eccentricity,
                periodS: model.scenario.period_s,
            },
            modelMetrics: model.metrics,
            traces: {},
        };

        for (const experiment of EXPERIMENTS) {
            const trace = buildTrace(model, scenarioInfo, experiment);
            const relPath = `data/traces/${scenarioInfo.id}_${experiment.id}.json`;
            writeJson(path.join(ROOT, "web", relPath), trace);
            scenarioEntry.traces[experiment.id] = {
                path: relPath,
                metrics: trace.metrics,
                source: trace.source,
            };
        }

        manifest.scenarios.push(scenarioEntry);
    }

    writeJson(path.join(OUT_DIR, "experiments.json"), manifest);
    console.log(`Wrote ${manifest.scenarios.length * EXPERIMENTS.length} traces to ${TRACE_DIR}`);
}

main();
