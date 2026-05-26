class KeplerLabApp {
    constructor() {
        this.boot = new BootSequence();
        this.board = null;
        this.liveBoard = null;
        this.surrogateBoard = null;
        this.surrogate = null;
        this.manifest = null;
        this.currentScenarioId = 'highly_elliptical';
        this.currentExperimentId = 'gradient_shock';
        this.currentTrace = null;
        this.liveTrace = null;
        this.finalTraceCache = new Map();
        this.currentIndex = 0;
        this.playing = false;
        this.lastFrame = 0;
        this.chartDpr = window.devicePixelRatio || 1;
        this.dom = {};
    }

    start() {
        this.boot.run(() => {
            this._init()
                .catch((error) => this._handleFatal(error));
        });
    }

    async _init() {
        this._bindDom();
        this._bindEvents();

        if (window.location.protocol === 'file:') {
            this._setStatus('SERVER REQUIRED', 'warning');
            this._setNotice(
                'The evidence JSON cannot be loaded from file://. Run node serve.js from the project root, then open http://localhost:8000/web/.'
            );
            this.board.render();
            return;
        }

        this._setStatus('LOADING DATA', 'loading');
        this.manifest = await this._loadJson(this._manifestCandidates());
        this._renderScenarioSelector();
        this._renderSteps();
        this._renderModelCard();
        await this._initSurrogate();
        await this._selectScenario(this.currentScenarioId);
        const requestedView = window.location.hash.replace('#', '');
        if (requestedView) this._setView(requestedView);
        this._setStatus('READY', 'ready');
        this._setNotice('');
        requestAnimationFrame((time) => this._loop(time));
    }

    _bindDom() {
        this.dom.statusText = document.getElementById('status-text');
        this.dom.statusBox = document.querySelector('.system-status');
        this.dom.notice = document.getElementById('model-notice');
        this.dom.provenanceTag = document.getElementById('provenance-tag');
        this.dom.scenarioSelect = document.getElementById('scenario-select');
        this.dom.steps = document.getElementById('autopsy-steps');
        this.dom.title = document.getElementById('experiment-title');
        this.dom.headline = document.getElementById('experiment-headline');
        this.dom.lesson = document.getElementById('experiment-lesson');
        this.dom.sourceBadge = document.getElementById('source-badge');
        this.dom.boardTitle = document.getElementById('board-title');
        this.dom.scrubber = document.getElementById('phase-scrubber');
        this.dom.scrubberReadout = document.getElementById('scrubber-readout');
        this.dom.playToggle = document.getElementById('play-toggle');
        this.dom.resetPhase = document.getElementById('reset-phase');
        this.dom.toggleError = document.getElementById('toggle-error');
        this.dom.toggleCollocation = document.getElementById('toggle-collocation');
        this.dom.metricPosition = document.getElementById('metric-position');
        this.dom.metricEnergy = document.getElementById('metric-energy');
        this.dom.metricMomentum = document.getElementById('metric-momentum');
        this.dom.metricInference = document.getElementById('metric-inference');
        this.dom.metricStatus = document.getElementById('metric-status');
        this.dom.errorChart = document.getElementById('chart-error');
        this.dom.energyChart = document.getElementById('chart-energy');
        this.dom.momentumChart = document.getElementById('chart-momentum');
        this.dom.liveScenario = document.getElementById('live-scenario');
        this.dom.liveParams = document.getElementById('live-params');
        this.dom.liveSpeedup = document.getElementById('live-speedup');
        this.dom.liveTitle = document.getElementById('live-title');
        this.dom.surrogateTitle = document.getElementById('surrogate-title');
        this.dom.surrogateBadge = document.getElementById('surrogate-bound-badge');
        this.dom.surrogateStatusCopy = document.getElementById('surrogate-status-copy');
        this.dom.surrogateQualityCopy =
            document.getElementById('surrogate-quality-copy') ||
            this.dom.surrogateStatusCopy;
        this.dom.surrogatePerigee = document.getElementById('surrogate-perigee');
        this.dom.surrogateApogee = document.getElementById('surrogate-apogee');
        this.dom.surrogatePerigeeReadout = document.getElementById('surrogate-perigee-readout');
        this.dom.surrogateApogeeReadout = document.getElementById('surrogate-apogee-readout');
        this.dom.surrogateEccentricity = document.getElementById('surrogate-eccentricity');
        this.dom.surrogateMeanError = document.getElementById('surrogate-mean-error');
        this.dom.surrogateMaxError = document.getElementById('surrogate-max-error');
        this.dom.surrogateDomain = document.getElementById('surrogate-domain-value');
        this.dom.latencyP50 = document.getElementById('latency-p50');
        this.dom.latencyP95 = document.getElementById('latency-p95');
        this.dom.latencyP99 = document.getElementById('latency-p99');
        this.dom.latencyJitter = document.getElementById('latency-jitter');
        this.dom.latencyVerlet = document.getElementById('latency-verlet');
        this.dom.latencySpeedup = document.getElementById('latency-speedup');
        this.dom.runLatency = document.getElementById('run-latency');
        this.dom.modelMetricsTable = document.getElementById('model-metrics-table');
        this.dom.surrogateMetricsTable = document.getElementById('surrogate-metrics-table');
        this.dom.tabs = Array.from(document.querySelectorAll('.mode-tab'));
        this.dom.views = Array.from(document.querySelectorAll('.view'));
        this.board = new KeplerBoard(document.getElementById('orbit-board'));
        this.liveBoard = new KeplerBoard(document.getElementById('live-board'));
        this.surrogateBoard = new KeplerBoard(document.getElementById('surrogate-board'));
    }

    _bindEvents() {
        this.dom.scenarioSelect.addEventListener('change', (event) => {
            this._selectScenario(event.target.value).catch((error) => this._handleFatal(error));
        });

        this.dom.scrubber.addEventListener('input', (event) => {
            this.playing = false;
            this.dom.playToggle.textContent = 'Play';
            this._setIndex(Number(event.target.value));
        });

        this.dom.playToggle.addEventListener('click', () => {
            this.playing = !this.playing;
            this.dom.playToggle.textContent = this.playing ? 'Pause' : 'Play';
        });

        this.dom.resetPhase.addEventListener('click', () => {
            this.playing = false;
            this.dom.playToggle.textContent = 'Play';
            this._setIndex(0);
        });

        this.dom.toggleError.addEventListener('change', () => this._syncBoardOverlays());
        this.dom.toggleCollocation.addEventListener('change', () => this._syncBoardOverlays());
        this.dom.surrogatePerigee.addEventListener('input', () => this._updateSurrogateFromControls());
        this.dom.surrogateApogee.addEventListener('input', () => this._updateSurrogateFromControls());
        this.dom.runLatency.addEventListener('click', () => this._runSurrogateLatency());

        for (const tab of this.dom.tabs) {
            tab.addEventListener('click', () => this._setView(tab.dataset.view));
        }

        window.addEventListener('resize', () => {
            this.board.resize();
            this.liveBoard.resize();
            this.surrogateBoard.resize();
            this._drawCharts();
        });
    }

    _manifestCandidates() {
        return [
            'data/experiments.json',
            'web/data/experiments.json',
            '/web/data/experiments.json',
        ];
    }

    _traceCandidates(path) {
        return [
            path,
            `web/${path}`,
            `/web/${path}`,
        ];
    }

    async _loadJson(candidates) {
        const errors = [];
        for (const url of candidates) {
            try {
                const response = await fetch(url, { cache: 'no-store' });
                if (!response.ok) {
                    throw new Error(`${response.status} ${response.statusText}`);
                }
                return await response.json();
            } catch (error) {
                errors.push(`${url}: ${error.message}`);
            }
        }
        throw new Error(`Could not load JSON. Tried ${errors.join(' | ')}`);
    }

    _renderScenarioSelector() {
        this.dom.scenarioSelect.innerHTML = '';
        for (const scenario of this.manifest.scenarios) {
            const option = document.createElement('option');
            option.value = scenario.id;
            option.textContent = scenario.label;
            this.dom.scenarioSelect.appendChild(option);
        }
        this.dom.scenarioSelect.value = this.currentScenarioId;
    }

    _renderSteps() {
        this.dom.steps.innerHTML = '';
        for (const experiment of this.manifest.experiments) {
            const button = document.createElement('button');
            button.className = 'autopsy-step';
            button.type = 'button';
            button.dataset.experiment = experiment.id;
            button.innerHTML = `
                <span class="step-index">${experiment.step}</span>
                <span class="step-name">${experiment.label}</span>
                <span class="step-status">${experiment.status}</span>
            `;
            button.addEventListener('click', () => {
                this._selectExperiment(experiment.id).catch((error) => this._handleFatal(error));
            });
            this.dom.steps.appendChild(button);
        }
        this._markActiveStep();
    }

    _renderModelCard() {
        if (!this.manifest) return;
        const rows = this.manifest.scenarios.map((scenario) => {
            const m = scenario.modelMetrics;
            const errorClass = m.position_error_mean_km > 10 ? 'warn' : 'pass';
            return `
                <div class="metrics-row ${errorClass}">
                    <span>${scenario.label}</span>
                    <strong>${this._fmt(m.position_error_mean_km, 3)} km mean | ${this._fmt(m.speedup_factor, 0)}x</strong>
                </div>
            `;
        });
        this.dom.modelMetricsTable.innerHTML = rows.join('');
    }

    async _initSurrogate() {
        this.surrogate = new SurrogateSolver(this.surrogateBoard);
        const result = await this.surrogate.init();
        this.dom.surrogateBadge.textContent = result.modelReady
            ? 'PARAMETRIC MODEL LOADED'
            : 'MODEL PENDING';
        this.dom.surrogateBadge.classList.toggle('measured', result.modelReady);
        this.dom.surrogateBadge.classList.toggle('pending', !result.modelReady);
        const arch = this.surrogate.model?.data?.architecture || {};
        this.dom.surrogateStatusCopy.textContent = result.modelReady
            ? (arch.output_mode === 'kepler_residual'
                ? 'The active surrogate solves Kepler anomaly first, then applies a bounded neural residual. The metrics below are measured against fresh Verlet truth.'
                : 'A legacy parametric export is active. It is preserved as evidence of the earlier high-e generalization failure.')
            : 'The generalized training/export path is wired. Train parametric_kepler.json to enable green PINN predictions here.';
        this._renderSurrogateModelCard(result.manifest);
        this._updateSurrogateFromControls();
    }

    _updateSurrogateFromControls() {
        if (!this.surrogate) return;
        const perigee = Number(this.dom.surrogatePerigee.value);
        const apogee = Math.max(perigee, Number(this.dom.surrogateApogee.value));
        if (apogee !== Number(this.dom.surrogateApogee.value)) {
            this.dom.surrogateApogee.value = String(apogee);
        }

        const trace = this.surrogate.buildTrace(perigee, apogee);
        const orbit = trace.orbit;
        const metrics = trace.metrics;
        const inDomain = this.surrogate.inTrainingDomain(orbit);

        this.dom.surrogatePerigeeReadout.textContent = `${this._fmt(perigee, 0)} km`;
        this.dom.surrogateApogeeReadout.textContent = `${this._fmt(apogee, 0)} km`;
        this.dom.surrogateTitle.textContent = `e=${this._fmt(orbit.eccentricity, 4)} | a=${this._fmt(orbit.semiMajorAxisM / 1000, 0)} km`;
        this.dom.surrogateEccentricity.textContent = this._fmt(orbit.eccentricity, 4);
        this.dom.surrogateMeanError.textContent = Number.isFinite(metrics.meanPositionErrorKm)
            ? `${this._fmt(metrics.meanPositionErrorKm, 2)} km`
            : '--';
        this.dom.surrogateMaxError.textContent = Number.isFinite(metrics.maxPositionErrorKm)
            ? `${this._fmt(metrics.maxPositionErrorKm, 2)} km`
            : '--';
        this.dom.surrogateDomain.textContent = inDomain ? 'inside' : 'outside';
        const status = this._surrogateStatus(metrics, inDomain);
        this.dom.surrogateBadge.textContent = status.label;
        this.dom.surrogateBadge.classList.toggle('pending', status.mode === 'pending' || status.mode === 'warn');
        this.dom.surrogateBadge.classList.toggle('measured', status.mode === 'pass');
        this.dom.surrogateBadge.classList.toggle('research', status.mode === 'fail');
        this.dom.surrogateQualityCopy.textContent = status.copy;
    }

    _surrogateStatus(metrics, inDomain) {
        if (!this.surrogate?.modelReady) {
            return {
                label: 'MODEL PENDING',
                mode: 'pending',
                copy: 'No parametric weights are loaded, so the board shows Velocity Verlet truth only.',
            };
        }
        if (!inDomain) {
            return {
                label: 'OUT OF DOMAIN',
                mode: 'warn',
                copy: 'This orbit sits outside the training parameter bounds. Treat the green trace as a stress test, not a claim.',
            };
        }
        if (!Number.isFinite(metrics.meanPositionErrorKm)) {
            return {
                label: 'NO METRIC',
                mode: 'warn',
                copy: 'The model loaded, but the live comparison did not produce a finite error metric.',
            };
        }
        if (metrics.meanPositionErrorKm < 10) {
            return {
                label: 'GENERALIZER',
                mode: 'pass',
                copy: `This orbit is inside the validated surrogate regime: ${this._fmt(metrics.meanPositionErrorKm, 2)} km mean error against fresh Verlet truth.`,
            };
        }
        if (metrics.meanPositionErrorKm < 75) {
            return {
                label: 'USABLE',
                mode: 'pass',
                copy: `This orbit is usable but not headline-grade: ${this._fmt(metrics.meanPositionErrorKm, 1)} km mean error against fresh Verlet truth.`,
            };
        }
        if (metrics.meanPositionErrorKm < 250) {
            return {
                label: 'PROTOTYPE',
                mode: 'warn',
                copy: `This is a prototype regime: ${this._fmt(metrics.meanPositionErrorKm, 0)} km mean error. Useful for showing the interface, not for the headline result.`,
            };
        }
        return {
            label: 'RESEARCH ONLY',
            mode: 'fail',
            copy: `This exposes the current generalization failure: ${this._fmt(metrics.meanPositionErrorKm, 0)} km mean error. The validated model for high-e is in Failure Lab.`,
        };
    }

    _renderSurrogateModelCard(manifest) {
        if (!this.dom.surrogateMetricsTable) return;
        if (!manifest || manifest.status !== 'model_ready') {
            this.dom.surrogateMetricsTable.innerHTML = `
                <div class="metrics-row warn">
                    <span>Parametric model</span>
                    <strong>pending export</strong>
                </div>
            `;
            return;
        }

        const cases =
            manifest.metrics?.traceCases ||
            manifest.metrics?.cases ||
            Object.entries(manifest.traces || {}).map(([id, trace]) => ({
                id,
                label: id.replace(/_/g, ' '),
                split: trace.split,
                position_error_mean_km: trace.metrics?.meanPositionErrorKm,
            }));

        const rows = cases.map((item) => {
            const mean = item.position_error_mean_km ?? item.meanPositionErrorKm;
            let cls = 'fail';
            if (item.split !== 'out_of_domain' && mean < 10) cls = 'pass';
            else if (item.split !== 'out_of_domain' && mean < 250) cls = 'warn';
            return `
                <div class="metrics-row ${cls}">
                    <span>${item.label}</span>
                    <strong>${this._fmt(mean, 1)} km mean | ${item.split || 'eval'}</strong>
                </div>
            `;
        });
        this.dom.surrogateMetricsTable.innerHTML = rows.join('');
    }

    _runSurrogateLatency() {
        if (!this.surrogate) return;
        const perigee = Number(this.dom.surrogatePerigee.value);
        const apogee = Math.max(perigee, Number(this.dom.surrogateApogee.value));
        const result = this.surrogate.benchmark(perigee, apogee);

        if (result.pinn) {
            this.dom.latencyP50.textContent = `${this._fmt(result.pinn.p50, 4)} ms`;
            this.dom.latencyP95.textContent = `${this._fmt(result.pinn.p95, 4)} ms`;
            this.dom.latencyP99.textContent = `${this._fmt(result.pinn.p99, 4)} ms`;
            this.dom.latencyJitter.textContent = `${this._fmt(result.pinn.jitter, 4)} ms`;
            this.dom.latencySpeedup.textContent = `${this._fmt(result.verletMs / Math.max(result.pinn.p50, 1e-6), 0)}x`;
        } else {
            this.dom.latencyP50.textContent = '--';
            this.dom.latencyP95.textContent = '--';
            this.dom.latencyP99.textContent = '--';
            this.dom.latencyJitter.textContent = '--';
            this.dom.latencySpeedup.textContent = '--';
        }
        this.dom.latencyVerlet.textContent = `${this._fmt(result.verletMs, 2)} ms`;
    }

    async _selectScenario(id) {
        this.currentScenarioId = id;
        this.dom.scenarioSelect.value = id;
        const scenario = this._scenario();
        const traceInfo = scenario.traces[this.currentExperimentId];
        if (!traceInfo) {
            this.currentExperimentId = this.manifest.experiments[0].id;
        }
        await this._loadCurrentTrace();
    }

    async _selectExperiment(id) {
        this.currentExperimentId = id;
        await this._loadCurrentTrace();
    }

    async _loadCurrentTrace() {
        const scenario = this._scenario();
        const experiment = this._experiment();
        const traceInfo = scenario.traces[experiment.id];
        if (!traceInfo) {
            throw new Error(`No trace for ${scenario.id}/${experiment.id}`);
        }

        this._setStatus('LOADING TRACE', 'loading');
        this.currentTrace = await this._loadJson(this._traceCandidates(traceInfo.path));
        await this._loadFinalTrace();
        this.currentIndex = 0;
        this.playing = false;
        this.dom.playToggle.textContent = 'Play';

        this.dom.scrubber.max = Math.max(0, this.currentTrace.series.length - 1);
        this.dom.scrubber.value = '0';
        this.board.setTrace(this.currentTrace);
        if (this.liveTrace) {
            this.liveBoard.setTrace(this.liveTrace);
            this.liveBoard.setOverlays({ showError: true, showCollocation: false });
        }
        this._syncBoardOverlays();
        this._markActiveStep();
        this._updateNarrative();
        this._setIndex(0);
        this._setStatus(this._statusTextForTrace(), this._statusModeForTrace());
    }

    _updateNarrative() {
        const scenario = this._scenario();
        const experiment = this._experiment();
        const source = this.currentTrace.source || experiment.source;
        const measured = source === 'measured_exported_model';

        this.dom.title.textContent = experiment.label;
        this.dom.headline.textContent = experiment.headline;
        this.dom.lesson.textContent = experiment.lesson;
        this.dom.boardTitle.textContent = `${scenario.label}: ${experiment.label}`;
        if (this.dom.provenanceTag) {
            this.dom.provenanceTag.textContent = source || 'unknown';
        }
        this.dom.sourceBadge.textContent = measured
            ? 'MEASURED EXPORTED WEIGHTS'
            : 'DIAGNOSTIC RECONSTRUCTION';
        this.dom.sourceBadge.classList.toggle('measured', measured);

        this.dom.liveScenario.textContent = scenario.label;
        this.dom.liveSpeedup.textContent = `${this._fmt(scenario.modelMetrics.speedup_factor, 0)}x`;
        this.dom.liveParams.textContent = '14,084';
        this.dom.liveTitle.textContent = `${scenario.label}: Final Model`;

        if (!measured) {
            this._setNotice(
                'Autopsy stages 01-04 are compact diagnostic reconstructions. The final step is measured from exported model weights.'
            );
        } else if (scenario.modelMetrics.position_error_mean_km > 100) {
            this._setNotice(
                `${scenario.label} is honest but still rough: ${this._fmt(scenario.modelMetrics.position_error_mean_km, 1)} km mean error. Longer high-e retraining is the next technical target.`
            );
        } else {
            this._setNotice('');
        }
    }

    _setIndex(index) {
        if (!this.currentTrace) return;
        const last = this.currentTrace.series.length - 1;
        this.currentIndex = Math.max(0, Math.min(last, Math.round(index)));
        this.dom.scrubber.value = String(this.currentIndex);
        this.board.setIndex(this.currentIndex);
        if (this.liveTrace) {
            const liveLast = this.liveTrace.series.length - 1;
            const liveIndex = Math.round((this.currentIndex / last) * liveLast);
            this.liveBoard.setIndex(liveIndex);
        }
        this._updateMetrics();
        this._drawCharts();
    }

    _updateMetrics() {
        if (!this.currentTrace) return;
        const point = this.currentTrace.series[this.currentIndex];
        const metrics = this.currentTrace.metrics;

        this.dom.scrubberReadout.textContent = point.orbit.toFixed(3);
        this.dom.metricPosition.textContent = `${this._fmt(point.err, 2)} km`;
        this.dom.metricEnergy.textContent = `${this._fmt(Math.abs(point.e), 3)} %`;
        this.dom.metricMomentum.textContent = `${this._fmt(Math.abs(point.h), 3)} %`;
        this.dom.metricInference.textContent = `${this._fmt(metrics.pinnInferenceMs, 3)} ms`;
        this.dom.metricStatus.textContent = this._metricStatus(metrics);
    }

    _metricStatus(metrics) {
        if (!metrics.measured) return 'autopsy';
        if (metrics.meanPositionErrorKm < 10) return 'flight-clean';
        if (metrics.meanPositionErrorKm < 120) return 'needs polish';
        return 'retrain';
    }

    _syncBoardOverlays() {
        if (!this.board) return;
        this.board.setOverlays({
            showError: this.dom.toggleError.checked,
            showCollocation: this.dom.toggleCollocation.checked,
        });
        if (this.liveBoard) {
            this.liveBoard.setOverlays({
                showError: true,
                showCollocation: false,
            });
        }
    }

    _drawCharts() {
        if (!this.currentTrace) return;
        this._drawChart(this.dom.errorChart, 'err', '#ef4444', 'km');
        this._drawChart(this.dom.energyChart, 'e', '#a78bfa', '%');
        this._drawChart(this.dom.momentumChart, 'h', '#fbbf24', '%');
    }

    _drawChart(canvas, key, color, unit) {
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(260, rect.width || canvas.clientWidth || 420);
        const height = Math.max(100, rect.height || canvas.clientHeight || 130);
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
        }

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const padX = 34 * dpr;
        const padY = 18 * dpr;
        const series = this.currentTrace.series;
        const values = series.map((point) => Math.abs(point[key]));
        const max = Math.max(...values, 1e-6);

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#050606';
        ctx.fillRect(0, 0, w, h);

        ctx.strokeStyle = 'rgba(100,112,107,0.26)';
        ctx.lineWidth = dpr;
        ctx.beginPath();
        ctx.moveTo(padX, padY);
        ctx.lineTo(padX, h - padY);
        ctx.lineTo(w - padX * 0.5, h - padY);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.7 * dpr;
        ctx.beginPath();
        values.forEach((value, index) => {
            const x = padX + (index / (values.length - 1)) * (w - padX * 1.5);
            const y = h - padY - (value / max) * (h - padY * 2);
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        const markerX = padX + (this.currentIndex / (values.length - 1)) * (w - padX * 1.5);
        ctx.strokeStyle = 'rgba(237,242,239,0.55)';
        ctx.beginPath();
        ctx.moveTo(markerX, padY);
        ctx.lineTo(markerX, h - padY);
        ctx.stroke();

        ctx.fillStyle = '#9ba7a2';
        ctx.font = `${10 * dpr}px JetBrains Mono, monospace`;
        ctx.fillText(`max ${this._fmt(max, key === 'err' ? 1 : 2)} ${unit}`, padX + 6 * dpr, padY + 11 * dpr);
    }

    _setView(view) {
        for (const tab of this.dom.tabs) {
            tab.classList.toggle('active', tab.dataset.view === view);
        }
        for (const panel of this.dom.views) {
            panel.classList.toggle('active', panel.id === `view-${view}`);
        }
        requestAnimationFrame(() => {
            this.board.resize();
            this.liveBoard.resize();
            this.surrogateBoard.resize();
            this._drawCharts();
        });
    }

    _loop(time) {
        if (this.playing && this.currentTrace) {
            const elapsed = Math.min(64, time - this.lastFrame);
            if (elapsed > 34) {
                const next = (this.currentIndex + 2) % this.currentTrace.series.length;
                this._setIndex(next);
                this.lastFrame = time;
            }
        } else {
            this.lastFrame = time;
        }
        requestAnimationFrame((nextTime) => this._loop(nextTime));
    }

    _markActiveStep() {
        for (const button of this.dom.steps.querySelectorAll('.autopsy-step')) {
            button.classList.toggle('active', button.dataset.experiment === this.currentExperimentId);
        }
    }

    _statusTextForTrace() {
        const metrics = this.currentTrace.metrics;
        if (!metrics.measured) return 'AUTOPSY';
        if (metrics.meanPositionErrorKm > 10) return 'RETRAIN';
        return 'MEASURED';
    }

    _statusModeForTrace() {
        const metrics = this.currentTrace.metrics;
        if (!metrics.measured) return 'warning';
        if (metrics.meanPositionErrorKm > 10) return 'warning';
        return 'ready';
    }

    async _loadFinalTrace() {
        const scenario = this._scenario();
        if (this.finalTraceCache.has(scenario.id)) {
            this.liveTrace = this.finalTraceCache.get(scenario.id);
            return;
        }

        const finalInfo = scenario.traces.final_model;
        if (!finalInfo) {
            this.liveTrace = null;
            return;
        }

        const trace = await this._loadJson(this._traceCandidates(finalInfo.path));
        this.finalTraceCache.set(scenario.id, trace);
        this.liveTrace = trace;
    }

    _setStatus(text, mode = 'ready') {
        this.dom.statusText.textContent = text;
        this.dom.statusBox.classList.remove('ready', 'loading', 'warning', 'error');
        this.dom.statusBox.classList.add(mode);
    }

    _setNotice(text) {
        this.dom.notice.textContent = text;
        this.dom.notice.classList.toggle('hidden', !text);
    }

    _handleFatal(error) {
        console.error(error);
        this._setStatus('LOAD ERROR', 'error');
        this._setNotice(error.message);
    }

    _scenario() {
        return this.manifest.scenarios.find((scenario) => scenario.id === this.currentScenarioId);
    }

    _experiment() {
        return this.manifest.experiments.find((experiment) => experiment.id === this.currentExperimentId);
    }

    _fmt(value, digits = 2) {
        if (!Number.isFinite(value)) return '--';
        return Number(value).toLocaleString(undefined, {
            minimumFractionDigits: 0,
            maximumFractionDigits: digits,
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new KeplerLabApp().start();
});
