const SurrogatePhysics = {
    G: 6.67430e-11,
    M_EARTH: 5.972e24,
    R_EARTH: 6.371e6,
};
SurrogatePhysics.MU = SurrogatePhysics.G * SurrogatePhysics.M_EARTH;

class ParametricKeplerModel {
    constructor() {
        this.loaded = false;
        this.data = null;
        this.numFreq = 10;
        this.featureType = 'fourier_time';
        this.outputMode = 'direct_state';
        this.residualScale = 1;
        this.keplerIterations = 0;
        this.conditionLayers = [];
        this.trunkLayers = [];
        this.bounds = null;
    }

    async load(candidates) {
        const errors = [];
        for (const url of candidates) {
            try {
                const response = await fetch(url, { cache: 'no-store' });
                if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
                this._parse(await response.json());
                this.loaded = true;
                return { ok: true, url };
            } catch (error) {
                errors.push(`${url}: ${error.message}`);
            }
        }
        return { ok: false, errors };
    }

    _parse(data) {
        this.data = data;
        this.numFreq = data.architecture.num_fourier_frequencies;
        this.featureType = data.architecture.feature_type || 'fourier_time';
        this.outputMode = data.architecture.output_mode || 'direct_state';
        this.residualScale = Number.isFinite(data.architecture.residual_scale)
            ? data.architecture.residual_scale
            : 1;
        this.keplerIterations = data.architecture.kepler_iterations || 0;
        this.bounds = data.parameter_bounds;
        this.conditionLayers = this._parseLayers(data.weights.condition_encoder);
        this.trunkLayers = this._parseLayers(data.weights.trunk);
    }

    _parseLayers(rawLayers) {
        return rawLayers.map((layer) => ({
            weight: layer.weight,
            bias: layer.bias,
            outDim: layer.bias.length,
            inDim: layer.weight[0].length,
        }));
    }

    forward(tNorm, eccentricity, semiMajorAxisM) {
        const featurePacket = this._featuresWithBaseline(tNorm, eccentricity);
        const paramFeatures = this._parameterFeatures(eccentricity, semiMajorAxisM);
        const condition = this._runLayers(paramFeatures, this.conditionLayers);
        const raw = this._runLayers([...featurePacket.features, ...condition], this.trunkLayers);
        if (this.outputMode !== 'kepler_residual') return raw;

        return featurePacket.baseline.map((value, index) => value + this.residualScale * raw[index]);
    }

    benchmark(sample) {
        if (!this.loaded) return null;
        const timings = [];
        const n = 600;
        for (let i = 0; i < n; i++) {
            const t = (i / n) * Math.PI * 2;
            const start = performance.now();
            this.forward(t, sample.eccentricity, sample.semiMajorAxisM);
            timings.push(performance.now() - start);
        }
        timings.sort((a, b) => a - b);
        const pick = (p) => timings[Math.min(timings.length - 1, Math.floor(p * timings.length))];
        return {
            p50: pick(0.50),
            p95: pick(0.95),
            p99: pick(0.99),
            jitter: pick(0.99) - pick(0.50),
        };
    }

    _fourier(tNorm) {
        const out = [];
        for (let k = 1; k <= this.numFreq; k++) {
            out.push(Math.sin(k * tNorm));
        }
        for (let k = 1; k <= this.numFreq; k++) {
            out.push(Math.cos(k * tNorm));
        }
        return out;
    }

    _featuresWithBaseline(tNorm, eccentricity) {
        if (this.featureType !== 'kepler_anomaly') {
            return { features: this._fourier(tNorm), baseline: null };
        }

        const info = this._keplerInfo(tNorm, eccentricity);
        const out = [];
        for (let k = 1; k <= this.numFreq; k++) out.push(Math.sin(k * tNorm));
        for (let k = 1; k <= this.numFreq; k++) out.push(Math.cos(k * tNorm));
        for (let k = 1; k <= this.numFreq; k++) out.push(Math.sin(k * info.eccentricAnomaly));
        for (let k = 1; k <= this.numFreq; k++) out.push(Math.cos(k * info.eccentricAnomaly));
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
        return { features: out, baseline: info.state };
    }

    _keplerState(tNorm, eccentricity) {
        return this._keplerInfo(tNorm, eccentricity).state;
    }

    _keplerInfo(tNorm, eccentricity) {
        const e = Math.min(0.95, Math.max(0, eccentricity));
        const eccentricAnomaly = this._solveEccentricAnomaly(tNorm, e);
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

    _solveEccentricAnomaly(meanAnomaly, eccentricity) {
        const e = eccentricity;
        const starterDen = Math.max(
            0.1,
            1 - Math.sin(meanAnomaly + e) + Math.sin(meanAnomaly),
        );
        let eccentricAnomaly = meanAnomaly + (e * Math.sin(meanAnomaly)) / starterDen;
        const iterations = Math.max(1, this.keplerIterations || 7);
        for (let i = 0; i < iterations; i++) {
            const residual = eccentricAnomaly - e * Math.sin(eccentricAnomaly) - meanAnomaly;
            const slope = Math.max(1e-5, 1 - e * Math.cos(eccentricAnomaly));
            eccentricAnomaly -= residual / slope;
        }
        return eccentricAnomaly;
    }

    _parameterFeatures(eccentricity, semiMajorAxisM) {
        const aBounds = this.bounds.semi_major_axis_m;
        const minLog = Math.log(aBounds[0]);
        const maxLog = Math.log(aBounds[1]);
        const aFeature = (2 * (Math.log(semiMajorAxisM) - minLog)) / (maxLog - minLog) - 1;
        return [eccentricity, aFeature];
    }

    _runLayers(input, layers) {
        let x = input;
        for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            const layer = layers[layerIndex];
            const applyTanh = layerIndex < layers.length - 1;
            const out = new Array(layer.outDim);
            for (let i = 0; i < layer.outDim; i++) {
                let sum = layer.bias[i];
                const row = layer.weight[i];
                for (let j = 0; j < layer.inDim; j++) sum += row[j] * x[j];
                out[i] = applyTanh ? Math.tanh(sum) : sum;
            }
            x = out;
        }
        return x;
    }
}

class SurrogateSolver {
    constructor(board) {
        this.board = board;
        this.model = new ParametricKeplerModel();
        this.manifest = null;
        this.trace = null;
        this.modelReady = false;
    }

    async init() {
        this.manifest = await this._loadManifest();
        const loadResult = await this.model.load([
            'models/parametric_kepler.json',
            '../models/parametric_kepler.json',
            '/models/parametric_kepler.json',
            '/web/models/parametric_kepler.json',
        ]);
        this.modelReady = loadResult.ok;
        return {
            modelReady: this.modelReady,
            manifest: this.manifest,
            loadResult,
        };
    }

    buildTrace(perigeeAltKm, apogeeAltKm, points = 720) {
        const orbit = this._orbitFromAltitudes(perigeeAltKm, apogeeAltKm);
        const truth = this._integrateTruth(orbit, points);
        const series = [];
        let totalError = 0;
        let maxError = 0;
        let errorCount = 0;

        for (let i = 0; i < points; i++) {
            const tNorm = (i / (points - 1)) * Math.PI * 2;
            const gt = truth[i];
            let pred = null;
            let err = null;
            let eDrift = null;
            let hDrift = null;

            if (this.modelReady) {
                const stateNorm = this.model.forward(tNorm, orbit.eccentricity, orbit.semiMajorAxisM);
                const predPhys = this._denorm(stateNorm, orbit);
                pred = [predPhys.x / 1000, predPhys.y / 1000];
                err = Math.hypot(predPhys.x - gt.x, predPhys.y - gt.y) / 1000;
                const energy0 = -0.5;
                const h0 = Math.sqrt(Math.max(0, 1 - orbit.eccentricity * orbit.eccentricity));
                const energy = this._energyNorm(stateNorm);
                const momentum = this._momentumNorm(stateNorm);
                eDrift = ((energy - energy0) / Math.abs(energy0)) * 100;
                hDrift = ((momentum - h0) / Math.abs(h0 || 1)) * 100;
                totalError += err;
                maxError = Math.max(maxError, err);
                errorCount++;
            }

            series.push({
                t: tNorm,
                orbit: tNorm / (Math.PI * 2),
                gt: [gt.x / 1000, gt.y / 1000],
                pred,
                err,
                e: eDrift,
                h: hDrift,
            });
        }

        this.trace = {
            scenario: 'parametric_kepler',
            experiment: this.modelReady ? 'parametric_model' : 'model_pending',
            label: this.modelReady ? 'Parametric PINN' : 'Model Pending',
            source: this.modelReady ? 'parametric_exported_model' : 'verlet_truth_only',
            orbit,
            series,
            metrics: {
                measured: this.modelReady,
                meanPositionErrorKm: errorCount ? totalError / errorCount : null,
                maxPositionErrorKm: errorCount ? maxError : null,
            },
        };
        this.board.setTrace(this.trace);
        this.board.setOverlays({ showError: this.modelReady, showCollocation: false });
        return this.trace;
    }

    benchmark(perigeeAltKm, apogeeAltKm) {
        const orbit = this._orbitFromAltitudes(perigeeAltKm, apogeeAltKm);
        const pinn = this.model.benchmark(orbit);
        const start = performance.now();
        this._integrateTruth(orbit, 2400);
        const verletMs = performance.now() - start;
        return { pinn, verletMs };
    }

    inTrainingDomain(orbit) {
        const bounds = this.modelReady ? this.model.bounds : this.manifest.parameterBounds;
        const eBounds = bounds.eccentricity;
        const aBounds = bounds.semi_major_axis_m;
        return (
            orbit.eccentricity >= eBounds[0] &&
            orbit.eccentricity <= eBounds[1] &&
            orbit.semiMajorAxisM >= aBounds[0] &&
            orbit.semiMajorAxisM <= aBounds[1]
        );
    }

    _loadManifest() {
        return fetch('data/surrogate.json', { cache: 'no-store' })
            .then((response) => {
                if (!response.ok) throw new Error('surrogate manifest missing');
                return response.json();
            })
            .catch(() => ({
                status: 'model_pending',
                parameterBounds: {
                    eccentricity: [0, 0.8],
                    semi_major_axis_m: [6500000, 45000000],
                },
            }));
    }

    _orbitFromAltitudes(perigeeAltKm, apogeeAltKm) {
        const rp = SurrogatePhysics.R_EARTH + perigeeAltKm * 1000;
        const ra = SurrogatePhysics.R_EARTH + Math.max(apogeeAltKm, perigeeAltKm) * 1000;
        const a = 0.5 * (rp + ra);
        const e = (ra - rp) / (ra + rp);
        const vScale = Math.sqrt(SurrogatePhysics.MU / a);
        const tScale = Math.sqrt((a * a * a) / SurrogatePhysics.MU);
        const period = 2 * Math.PI * tScale;
        const vp = Math.sqrt(SurrogatePhysics.MU * (2 / rp - 1 / a));
        return {
            perigeeAltKm,
            apogeeAltKm: Math.max(apogeeAltKm, perigeeAltKm),
            eccentricity: e,
            semiMajorAxisM: a,
            xScaleM: a,
            vScaleMs: vScale,
            periodS: period,
            initialState: [rp, 0, 0, vp],
        };
    }

    _integrateTruth(orbit, points) {
        const steps = Math.max(2000, points * 6);
        const dt = orbit.periodS / steps;
        const sampleEvery = steps / (points - 1);
        const out = [];
        let x = orbit.initialState[0];
        let y = orbit.initialState[1];
        let vx = orbit.initialState[2];
        let vy = orbit.initialState[3];
        let a = this._acceleration(x, y);

        for (let step = 0; step <= steps; step++) {
            const sampleIndex = Math.round(out.length * sampleEvery);
            if (step === sampleIndex && out.length < points) {
                out.push({ x, y, vx, vy });
            }

            const xNew = x + vx * dt + 0.5 * a.ax * dt * dt;
            const yNew = y + vy * dt + 0.5 * a.ay * dt * dt;
            const aNew = this._acceleration(xNew, yNew);
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

    _acceleration(x, y) {
        const r2 = x * x + y * y;
        const r = Math.sqrt(r2);
        const r3 = r2 * r;
        return {
            ax: -SurrogatePhysics.MU * x / r3,
            ay: -SurrogatePhysics.MU * y / r3,
        };
    }

    _denorm(stateNorm, orbit) {
        return {
            x: stateNorm[0] * orbit.xScaleM,
            y: stateNorm[1] * orbit.xScaleM,
            vx: stateNorm[2] * orbit.vScaleMs,
            vy: stateNorm[3] * orbit.vScaleMs,
        };
    }

    _energyNorm(state) {
        const [x, y, vx, vy] = state;
        const r = Math.sqrt(x * x + y * y + 1e-6);
        return 0.5 * (vx * vx + vy * vy) - 1 / r;
    }

    _momentumNorm(state) {
        return state[0] * state[3] - state[1] * state[2];
    }
}
