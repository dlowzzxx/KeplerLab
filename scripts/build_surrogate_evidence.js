const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(process.env.PINN_ROOT || path.resolve(__dirname, ".."));
const MODEL_PATH = path.join(ROOT, "models", "parametric_kepler.json");
const OUT_DIR = path.join(ROOT, "web", "data");
const TRACE_DIR = path.join(OUT_DIR, "traces");

const G = 6.67430e-11;
const M_EARTH = 5.972e24;
const R_EARTH = 6.371e6;
const MU = G * M_EARTH;

const DEFAULT_BOUNDS = {
    eccentricity: [0, 0.8],
    eccentricity_eval: [0, 0.9],
    semi_major_axis_m: [6.6e6, 4.5e7],
};

const CASES = [
    { id: "seen_low_e", label: "Seen low-e", eccentricity: 0.1, semiMajorAxisM: 9000000, split: "in_domain" },
    { id: "seen_mid_e", label: "Seen mid-e", eccentricity: 0.45, semiMajorAxisM: 16000000, split: "in_domain" },
    { id: "seen_high_e", label: "Seen high-e", eccentricity: 0.72, semiMajorAxisM: 24414000, split: "in_domain" },
    { id: "ood_extreme_e", label: "OOD extreme-e", eccentricity: 0.87, semiMajorAxisM: 24414000, split: "out_of_domain" },
];

function ensureDirs() {
    fs.mkdirSync(TRACE_DIR, { recursive: true });
}

function writeJson(file, data) {
    fs.writeFileSync(file, `${JSON.stringify(data)}\n`);
}

function round(value, digits = 6) {
    if (!Number.isFinite(value)) return null;
    return Number(value.toFixed(digits));
}

function orbitFromCase(testCase) {
    const a = testCase.semiMajorAxisM;
    const e = testCase.eccentricity;
    const rp = a * (1 - e);
    const ra = a * (1 + e);
    const vp = Math.sqrt(MU * (2 / rp - 1 / a));
    const period = 2 * Math.PI * Math.sqrt((a * a * a) / MU);
    return {
        eccentricity: e,
        semiMajorAxisM: a,
        perigeeAltKm: rp / 1000 - R_EARTH / 1000,
        apogeeAltKm: ra / 1000 - R_EARTH / 1000,
        periodS: period,
        xScaleM: a,
        vScaleMs: Math.sqrt(MU / a),
        initialState: [rp, 0, 0, vp],
    };
}

function acceleration(x, y) {
    const r2 = x * x + y * y;
    const r = Math.sqrt(r2);
    const r3 = r2 * r;
    return { ax: -MU * x / r3, ay: -MU * y / r3 };
}

function integrateTruth(orbit, points = 720) {
    const steps = Math.max(3000, points * 6);
    const dt = orbit.periodS / steps;
    const out = [];
    let x = orbit.initialState[0];
    let y = orbit.initialState[1];
    let vx = orbit.initialState[2];
    let vy = orbit.initialState[3];
    let a = acceleration(x, y);

    for (let step = 0; step <= steps; step++) {
        const target = Math.round((out.length / (points - 1)) * steps);
        if (step === target && out.length < points) out.push({ x, y, vx, vy });

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
    }

    while (out.length < points) out.push({ x, y, vx, vy });
    return out;
}

function fourier(t, n) {
    const out = [];
    for (let k = 1; k <= n; k++) out.push(Math.sin(k * t));
    for (let k = 1; k <= n; k++) out.push(Math.cos(k * t));
    return out;
}

function solveEccentricAnomaly(meanAnomaly, eccentricity, iterations = 7) {
    const e = Math.min(0.95, Math.max(0, eccentricity));
    const starterDen = Math.max(
        0.1,
        1 - Math.sin(meanAnomaly + e) + Math.sin(meanAnomaly),
    );
    let eccentricAnomaly = meanAnomaly + (e * Math.sin(meanAnomaly)) / starterDen;
    for (let i = 0; i < Math.max(1, iterations); i++) {
        const residual = eccentricAnomaly - e * Math.sin(eccentricAnomaly) - meanAnomaly;
        const slope = Math.max(1e-5, 1 - e * Math.cos(eccentricAnomaly));
        eccentricAnomaly -= residual / slope;
    }
    return eccentricAnomaly;
}

function keplerInfo(tNorm, eccentricity, iterations = 7) {
    const e = Math.min(0.95, Math.max(0, eccentricity));
    const eccentricAnomaly = solveEccentricAnomaly(tNorm, e, iterations);
    const beta = Math.sqrt(Math.max(1e-8, 1 - e * e));
    const sinE = Math.sin(eccentricAnomaly);
    const cosE = Math.cos(eccentricAnomaly);
    const denom = Math.max(1e-5, 1 - e * cosE);
    return {
        eccentricity: e,
        eccentricAnomaly,
        beta,
        denom,
        state: [
            cosE - e,
            beta * sinE,
            -sinE / denom,
            beta * cosE / denom,
        ],
    };
}

function timeFeatures(model, tNorm, eccentricity) {
    const architecture = model.architecture || {};
    const n = architecture.num_fourier_frequencies;
    if (architecture.feature_type !== "kepler_anomaly") return fourier(tNorm, n);

    const info = keplerInfo(tNorm, eccentricity, architecture.kepler_iterations || 7);
    const out = [];
    for (let k = 1; k <= n; k++) out.push(Math.sin(k * tNorm));
    for (let k = 1; k <= n; k++) out.push(Math.cos(k * tNorm));
    for (let k = 1; k <= n; k++) out.push(Math.sin(k * info.eccentricAnomaly));
    for (let k = 1; k <= n; k++) out.push(Math.cos(k * info.eccentricAnomaly));
    out.push(
        info.state[0],
        info.state[1],
        info.state[2],
        info.state[3],
        info.denom,
        1 / info.denom,
        info.beta,
        info.eccentricity,
    );
    return out;
}

function runLayers(input, layers) {
    let x = input;
    for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        const layer = layers[layerIndex];
        const tanh = layerIndex < layers.length - 1;
        const out = new Array(layer.bias.length);
        for (let i = 0; i < layer.bias.length; i++) {
            let sum = layer.bias[i];
            for (let j = 0; j < layer.weight[i].length; j++) sum += layer.weight[i][j] * x[j];
            out[i] = tanh ? Math.tanh(sum) : sum;
        }
        x = out;
    }
    return x;
}

function aFeature(model, semiMajorAxisM) {
    const bounds = model.parameter_bounds.semi_major_axis_m;
    const lo = Math.log(bounds[0]);
    const hi = Math.log(bounds[1]);
    return 2 * (Math.log(semiMajorAxisM) - lo) / (hi - lo) - 1;
}

function predict(model, tNorm, orbit) {
    const time = timeFeatures(model, tNorm, orbit.eccentricity);
    const params = [orbit.eccentricity, aFeature(model, orbit.semiMajorAxisM)];
    const condition = runLayers(params, model.weights.condition_encoder);
    const raw = runLayers([...time, ...condition], model.weights.trunk);
    if (model.architecture.output_mode !== "kepler_residual") return raw;

    const baseline = keplerInfo(
        tNorm,
        orbit.eccentricity,
        model.architecture.kepler_iterations || 7,
    ).state;
    const residualScale = Number.isFinite(model.architecture.residual_scale)
        ? model.architecture.residual_scale
        : 1;
    return baseline.map((value, index) => value + residualScale * raw[index]);
}

function denorm(state, orbit) {
    return {
        x: state[0] * orbit.xScaleM,
        y: state[1] * orbit.xScaleM,
        vx: state[2] * orbit.vScaleMs,
        vy: state[3] * orbit.vScaleMs,
    };
}

function energyNorm(state) {
    const [x, y, vx, vy] = state;
    const r = Math.sqrt(x * x + y * y + 1e-6);
    return 0.5 * (vx * vx + vy * vy) - 1 / r;
}

function momentumNorm(state) {
    return state[0] * state[3] - state[1] * state[2];
}

function buildTrace(model, testCase) {
    const orbit = orbitFromCase(testCase);
    const truth = integrateTruth(orbit);
    const initialEnergy = -0.5;
    const initialMomentum = Math.sqrt(Math.max(0, 1 - orbit.eccentricity * orbit.eccentricity));
    const series = [];

    for (let i = 0; i < truth.length; i++) {
        const tNorm = (i / (truth.length - 1)) * 2 * Math.PI;
        const stateNorm = predict(model, tNorm, orbit);
        const pred = denorm(stateNorm, orbit);
        const err = Math.hypot(pred.x - truth[i].x, pred.y - truth[i].y) / 1000;
        series.push({
            t: round(tNorm, 5),
            orbit: round(tNorm / (2 * Math.PI), 5),
            gt: [round(truth[i].x / 1000, 3), round(truth[i].y / 1000, 3)],
            pred: [round(pred.x / 1000, 3), round(pred.y / 1000, 3)],
            err: round(err, 4),
            e: round(((energyNorm(stateNorm) - initialEnergy) / Math.abs(initialEnergy)) * 100, 5),
            h: round(((momentumNorm(stateNorm) - initialMomentum) / Math.abs(initialMomentum || 1)) * 100, 5),
        });
    }

    const errors = series.map((p) => p.err);
    const energy = series.map((p) => Math.abs(p.e));
    const momentum = series.map((p) => Math.abs(p.h));
    return {
        caseId: testCase.id,
        label: testCase.label,
        split: testCase.split,
        source: "parametric_exported_model",
        orbit,
        series,
        metrics: {
            meanPositionErrorKm: round(errors.reduce((a, b) => a + b, 0) / errors.length, 4),
            maxPositionErrorKm: round(Math.max(...errors), 4),
            maxEnergyDriftPct: round(Math.max(...energy), 6),
            maxMomentumDriftPct: round(Math.max(...momentum), 6),
        },
    };
}

function pendingManifest() {
    return {
        generatedAt: new Date().toISOString(),
        status: "model_pending",
        modelPath: "models/parametric_kepler.json",
        parameterBounds: DEFAULT_BOUNDS,
        validationCases: CASES,
        traces: {},
        command: "python training/train_parametric.py --epochs 20000 --export",
    };
}

function main() {
    ensureDirs();
    if (!fs.existsSync(MODEL_PATH)) {
        writeJson(path.join(OUT_DIR, "surrogate.json"), pendingManifest());
        console.log("Parametric model pending; wrote web/data/surrogate.json");
        return;
    }

    const model = JSON.parse(fs.readFileSync(MODEL_PATH, "utf8"));
    const manifest = {
        generatedAt: new Date().toISOString(),
        status: "model_ready",
        modelPath: "models/parametric_kepler.json",
        modelType: model.model_type || "parametric_kepler_surrogate",
        architecture: {
            featureType: model.architecture?.feature_type || "fourier_time",
            outputMode: model.architecture?.output_mode || "direct_state",
            residualScale: model.architecture?.residual_scale ?? null,
            keplerIterations: model.architecture?.kepler_iterations ?? null,
        },
        parameterBounds: model.parameter_bounds,
        validationCases: CASES,
        traces: {},
        metrics: {
            source: "browser_trace_replay",
            traceCases: [],
        },
        trainingMetrics: model.metrics || {},
    };

    for (const testCase of CASES) {
        const trace = buildTrace(model, testCase);
        const relPath = `data/traces/parametric_${testCase.id}.json`;
        writeJson(path.join(ROOT, "web", relPath), trace);
        manifest.metrics.traceCases.push({
            id: testCase.id,
            label: testCase.label,
            split: testCase.split,
            eccentricity: testCase.eccentricity,
            position_error_mean_km: trace.metrics.meanPositionErrorKm,
            position_error_max_km: trace.metrics.maxPositionErrorKm,
            energy_drift_max_pct: trace.metrics.maxEnergyDriftPct,
            momentum_drift_max_pct: trace.metrics.maxMomentumDriftPct,
        });
        manifest.traces[testCase.id] = {
            path: relPath,
            metrics: trace.metrics,
            split: testCase.split,
        };
    }

    const inDomain = manifest.metrics.traceCases.filter((item) => item.split === "in_domain");
    const outOfDomain = manifest.metrics.traceCases.filter((item) => item.split === "out_of_domain");
    manifest.metrics.in_domain_mean_error_km = round(
        inDomain.reduce((sum, item) => sum + item.position_error_mean_km, 0) / Math.max(1, inDomain.length),
        4,
    );
    manifest.metrics.out_of_domain_mean_error_km = round(
        outOfDomain.reduce((sum, item) => sum + item.position_error_mean_km, 0) / Math.max(1, outOfDomain.length),
        4,
    );
    const highE = manifest.metrics.traceCases.find((item) => item.id === "seen_high_e");
    manifest.quality = {
        label: highE && highE.position_error_mean_km < 100
            ? "prototype_generalizer"
            : "research_only",
        note: highE && highE.position_error_mean_km < 100
            ? "Hybrid parametric model is usable for the listed in-domain validation cases."
            : "Parametric model is intentionally exposed as a research path; high-e generalization is not solved yet.",
    };

    writeJson(path.join(OUT_DIR, "surrogate.json"), manifest);
    console.log(`Wrote ${CASES.length} parametric surrogate traces`);
}

main();
