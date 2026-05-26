const fs = require('fs');
const path = require('path');

const root = path.resolve(process.env.PINN_ROOT || path.join(__dirname, '..'));
const manifestPath = path.join(root, 'web', 'data', 'experiments.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const surrogatePath = path.join(root, 'web', 'data', 'surrogate.json');

let hardFailures = 0;
let warnings = 0;

const FINAL_THRESHOLDS = {
    circular_leo: {
        meanKm: 1,
        maxKm: 5,
        energyPct: 0.1,
        momentumPct: 0.05,
    },
    elliptical: {
        meanKm: 1,
        maxKm: 5,
        energyPct: 0.1,
        momentumPct: 0.05,
    },
    highly_elliptical: {
        meanKm: 10,
        maxKm: 50,
        energyPct: 1.0,
        momentumPct: 0.25,
    },
};

function fail(message) {
    hardFailures += 1;
    console.error(`FAIL ${message}`);
}

function warn(message) {
    warnings += 1;
    console.warn(`WARN ${message}`);
}

function ok(message) {
    console.log(`OK   ${message}`);
}

function isFinitePoint(point) {
    return (
        Array.isArray(point.gt) &&
        Array.isArray(point.pred) &&
        point.gt.every(Number.isFinite) &&
        point.pred.every(Number.isFinite) &&
        Number.isFinite(point.err) &&
        Number.isFinite(point.e) &&
        Number.isFinite(point.h)
    );
}

for (const scenario of manifest.scenarios) {
    const traces = {};
    for (const [experimentId, traceInfo] of Object.entries(scenario.traces)) {
        const tracePath = path.join(root, 'web', traceInfo.path);
        const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
        traces[experimentId] = trace;

        if (!trace.series?.length) {
            fail(`${traceInfo.path} has no series points`);
            continue;
        }
        const badPoint = trace.series.find((point) => !isFinitePoint(point));
        if (badPoint) fail(`${traceInfo.path} contains nan/inf or malformed point data`);

        const uniform = trace.collocation?.uniform?.length || 0;
        const focused = trace.collocation?.focused?.length || 0;
        const total = uniform + focused;
        if (total > 0) {
            const ratio = focused / total;
            if (Math.abs(ratio - 0.5) > 0.05) {
                fail(`${traceInfo.path} collocation focus ratio is ${ratio.toFixed(3)}, expected about 0.5`);
            }
        }
    }

    const baseline = traces.gradient_shock?.metrics?.meanPositionErrorKm;
    const final = traces.final_model?.metrics?.meanPositionErrorKm;
    if (Number.isFinite(baseline) && Number.isFinite(final) && final < baseline) {
        ok(`${scenario.label}: final improves mean error (${final.toFixed(2)} km vs ${baseline.toFixed(2)} km)`);
    } else {
        fail(`${scenario.label}: final model does not improve over baseline`);
    }

    const threshold = FINAL_THRESHOLDS[scenario.id];
    const m = scenario.modelMetrics;
    if (threshold) {
        const failuresBeforeThreshold = hardFailures;
        if (m.position_error_mean_km > threshold.meanKm) {
            fail(`${scenario.label}: mean error ${m.position_error_mean_km} km exceeds ${threshold.meanKm} km gate`);
        }
        if (m.position_error_max_km > threshold.maxKm) {
            fail(`${scenario.label}: max error ${m.position_error_max_km} km exceeds ${threshold.maxKm} km gate`);
        }
        if (m.energy_drift_max_pct > threshold.energyPct) {
            fail(`${scenario.label}: energy drift ${m.energy_drift_max_pct}% exceeds ${threshold.energyPct}% gate`);
        }
        if (m.momentum_drift_max_pct > threshold.momentumPct) {
            fail(`${scenario.label}: momentum drift ${m.momentum_drift_max_pct}% exceeds ${threshold.momentumPct}% gate`);
        }
        if (hardFailures === failuresBeforeThreshold) {
            ok(`${scenario.label}: final metrics pass quality gates`);
        }
    }
}

if (hardFailures > 0) {
    console.error(`Evidence check failed with ${hardFailures} hard failure(s), ${warnings} warning(s).`);
    process.exit(1);
}

if (fs.existsSync(surrogatePath)) {
    const surrogate = JSON.parse(fs.readFileSync(surrogatePath, 'utf8'));
    if (!['model_pending', 'model_ready'].includes(surrogate.status)) {
        fail(`surrogate.json has invalid status: ${surrogate.status}`);
    } else {
        ok(`Surrogate manifest status: ${surrogate.status}`);
    }
    if (surrogate.status === 'model_ready') {
        const traceCases = surrogate.metrics?.traceCases || [];
        for (const traceInfo of Object.values(surrogate.traces || {})) {
            const tracePath = path.join(root, 'web', traceInfo.path);
            const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
            if (!trace.series?.length) fail(`${traceInfo.path} has no series points`);
            const badPoint = trace.series.find((point) => !isFinitePoint(point));
            if (badPoint) fail(`${traceInfo.path} contains malformed point data`);

            const caseMetric = traceCases.find((item) => item.id === trace.caseId);
            if (caseMetric) {
                const delta = Math.abs(caseMetric.position_error_mean_km - trace.metrics.meanPositionErrorKm);
                if (delta > 1e-4) {
                    fail(`${traceInfo.path} mean metric disagrees with surrogate manifest`);
                }
            }
        }

        const highE = traceCases.find((item) => item.id === 'seen_high_e');
        if (highE && highE.position_error_mean_km > 100) {
            if (surrogate.quality?.label === 'research_only') {
                ok('Surrogate quality gate documents high-e generalization as research-only');
            } else {
                fail('Parametric surrogate high-e error is high but not labeled research-only');
            }
        } else if (highE && highE.position_error_mean_km <= 100) {
            if (surrogate.quality?.label === 'prototype_generalizer') {
                ok(`Surrogate high-e gate passes (${highE.position_error_mean_km.toFixed(2)} km mean)`);
            } else {
                fail('Parametric surrogate passes high-e gate but is not labeled prototype_generalizer');
            }
        }
    }
} else {
    warn('web/data/surrogate.json is missing; run node scripts/build_surrogate_evidence.js');
}

if (hardFailures > 0) {
    console.error(`Evidence check failed with ${hardFailures} hard failure(s), ${warnings} warning(s).`);
    process.exit(1);
}

console.log(`Evidence check passed with ${warnings} warning(s).`);
