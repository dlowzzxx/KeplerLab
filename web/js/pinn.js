/**
 * pinn.js - PINN inference engine (pure JavaScript)
 *
 * Implements the MLP forward pass with Fourier feature embedding.
 * No ML framework required - just matrix multiplication and tanh.
 *
 * Architecture (loaded from JSON):
 *   t_norm -> Fourier Features (20-dim) -> [Linear->Tanh]x4 -> Linear -> [x,y,vx,vy]
 *
 * @author PINN Orbital Dynamics
 */

class PINNModel {
    constructor() {
        this.loaded = false;
        this.weights = [];    // Array of {w: Float64Array[], b: Float64Array}
        this.numFreq = 10;
        this.inputDim = 20;   // 2 * numFreq (sin + cos, no raw t)
        this.hiddenDim = 64;
        this.outputDim = 4;
        this.norm = { tScale: 1, xScale: 1, vScale: 1 };
        this.scenario = null;
        this.metrics = null;
        this.trainingHistory = null;
    }

    /**
     * Load model weights from a JSON file.
     * @param {string} url - Path to the model JSON file
     * @returns {Promise<void>}
     */
    async loadModel(url) {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`Failed to load model: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();

        // Parse architecture
        const arch = data.architecture;
        this.numFreq = arch.num_fourier_frequencies;
        this.inputDim = arch.input_dim;
        this.hiddenDim = arch.hidden_dim;
        this.outputDim = arch.output_dim;

        // Parse normalization scales
        this.norm = {
            tScale: data.normalization.t_scale,
            xScale: data.normalization.x_scale,
            vScale: data.normalization.v_scale,
        };

        // Parse scenario metadata
        this.scenario = data.scenario;
        this.metrics = data.metrics;
        this.trainingHistory = data.training_history;

        // Parse weights - convert nested arrays to typed arrays for speed
        this.weights = [];
        let layerIdx = 0;
        while (data.weights[`layer_${layerIdx}`] !== undefined) {
            const layer = data.weights[`layer_${layerIdx}`];
            const w = layer.weight;  // 2D array [out_dim][in_dim]
            const b = layer.bias;    // 1D array [out_dim]
            this.weights.push({
                w: w,                // Keep as nested arrays for simplicity
                b: new Float64Array(b),
                outDim: w.length,
                inDim: w[0].length,
            });
            layerIdx++;
        }

        this.loaded = true;
        console.log(`PINN model loaded: ${layerIdx} layers, ` +
                    `${this.countParameters()} parameters`);
    }

    /**
     * Count total trainable parameters.
     * @returns {number}
     */
    countParameters() {
        let total = 0;
        for (const layer of this.weights) {
            total += layer.outDim * layer.inDim + layer.outDim;
        }
        return total;
    }

    /**
     * Compute Fourier feature embedding for normalized time.
     *
     * Uses integer harmonics: [sin(t), cos(t), sin(2t), cos(2t), ..., sin(Nt), cos(Nt)]
     * This naturally captures the periodic structure of Keplerian orbits.
     *
     * @param {number} tNorm - Normalized time (one orbit = 2pi)
     * @returns {Float64Array} Feature vector of length 2*numFreq
     */
    fourierFeatures(tNorm) {
        const features = new Float64Array(this.inputDim);
        for (let k = 0; k < this.numFreq; k++) {
            const freq = k + 1;  // Integer harmonics: 1, 2, 3, ..., N
            const arg = freq * tNorm;
            features[k] = Math.sin(arg);
            features[k + this.numFreq] = Math.cos(arg);
        }
        return features;
    }

    /**
     * Single-layer forward pass: output = activation(W * input + b)
     *
     * @param {Float64Array} input - Input vector
     * @param {object} layer - {w, b, outDim, inDim}
     * @param {boolean} applyTanh - Whether to apply tanh activation
     * @returns {Float64Array} Output vector
     */
    linearForward(input, layer, applyTanh) {
        const output = new Float64Array(layer.outDim);
        for (let i = 0; i < layer.outDim; i++) {
            let sum = layer.b[i];
            const wi = layer.w[i];
            for (let j = 0; j < layer.inDim; j++) {
                sum += wi[j] * input[j];
            }
            output[i] = applyTanh ? Math.tanh(sum) : sum;
        }
        return output;
    }

    /**
     * Full MLP forward pass: time -> state.
     *
     * @param {number} tNorm - Normalized time
     * @returns {Float64Array} [x_norm, y_norm, vx_norm, vy_norm]
     */
    forward(tNorm) {
        let x = this.fourierFeatures(tNorm);

        // Hidden layers with tanh activation
        const nHidden = this.weights.length - 1;
        for (let i = 0; i < nHidden; i++) {
            x = this.linearForward(x, this.weights[i], true);
        }

        // Output layer - no activation
        x = this.linearForward(x, this.weights[nHidden], false);
        return x;
    }

    /**
     * Predict state at a physical time value.
     * Handles normalization/denormalization.
     *
     * @param {number} tPhysical - Time in seconds
     * @returns {{x: number, y: number, vx: number, vy: number}} Physical state
     */
    predict(tPhysical) {
        const tNorm = tPhysical / this.norm.tScale;
        const stateNorm = this.forward(tNorm);
        return {
            x:  stateNorm[0] * this.norm.xScale,
            y:  stateNorm[1] * this.norm.xScale,
            vx: stateNorm[2] * this.norm.vScale,
            vy: stateNorm[3] * this.norm.vScale,
        };
    }

    /**
     * Predict normalized state (for conservation law computation).
     *
     * @param {number} tNorm - Normalized time
     * @returns {{x: number, y: number, vx: number, vy: number}} Normalized state
     */
    predictNormalized(tNorm) {
        const s = this.forward(tNorm);
        return { x: s[0], y: s[1], vx: s[2], vy: s[3] };
    }

    /**
     * Predict full trajectory (for initial overlay computation).
     *
     * @param {number} tStart - Start time (normalized)
     * @param {number} tEnd - End time (normalized)
     * @param {number} nPoints - Number of points
     * @returns {Array<{x: number, y: number, vx: number, vy: number}>} Trajectory
     */
    predictTrajectory(tStart, tEnd, nPoints) {
        const trajectory = [];
        const dt = (tEnd - tStart) / (nPoints - 1);
        for (let i = 0; i < nPoints; i++) {
            const tNorm = tStart + i * dt;
            const s = this.forward(tNorm);
            trajectory.push({
                x:  s[0] * this.norm.xScale,
                y:  s[1] * this.norm.xScale,
                vx: s[2] * this.norm.vScale,
                vy: s[3] * this.norm.vScale,
                tNorm: tNorm,
            });
        }
        return trajectory;
    }
}
