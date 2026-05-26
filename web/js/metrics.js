/**
 * metrics.js - Real-time metrics charts for the PINN dashboard.
 *
 * Draws mini line charts on canvas elements showing:
 *   - Position error between PINN and Verlet
 *   - Energy conservation drift
 *   - Angular momentum drift
 */

class MetricsChart {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {string} color - Line color (CSS)
     * @param {string} unit - Unit label ('%', 'km', etc.)
     * @param {boolean} symmetric - If true, Y axis is symmetric around 0
     */
    constructor(canvas, color, unit, symmetric = false) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.color = color;
        this.unit = unit;
        this.symmetric = symmetric;

        this.data = [];
        this.maxPoints = 500;

        // Handle DPI
        this.dpr = window.devicePixelRatio || 1;
        this.resize();
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * this.dpr;
        this.canvas.height = rect.height * this.dpr;
        this.width = rect.width;
        this.height = rect.height;
    }

    /**
     * Add a data point.
     * @param {number} value
     */
    push(value) {
        this.data.push(value);
        if (this.data.length > this.maxPoints) {
            this.data.shift();
        }
    }

    /** Clear all data. */
    clear() {
        this.data = [];
    }

    /** Render the chart. */
    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const padding = 4 * this.dpr;

        // Background
        ctx.fillStyle = 'rgba(10, 15, 26, 0.9)';
        ctx.fillRect(0, 0, w, h);

        if (this.data.length < 2) return;

        // Compute Y range
        let yMin = Infinity, yMax = -Infinity;
        for (const v of this.data) {
            if (v < yMin) yMin = v;
            if (v > yMax) yMax = v;
        }

        // Add padding to range
        const yRange = yMax - yMin || 1;
        yMin -= yRange * 0.1;
        yMax += yRange * 0.1;

        if (this.symmetric) {
            const absMax = Math.max(Math.abs(yMin), Math.abs(yMax));
            yMin = -absMax;
            yMax = absMax;
        }

        const plotW = w - 2 * padding;
        const plotH = h - 2 * padding;

        // Zero line
        if (this.symmetric || (yMin < 0 && yMax > 0)) {
            const zeroY = padding + plotH * (1 - (0 - yMin) / (yMax - yMin));
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            ctx.lineWidth = 0.5 * this.dpr;
            ctx.setLineDash([2 * this.dpr, 3 * this.dpr]);
            ctx.beginPath();
            ctx.moveTo(padding, zeroY);
            ctx.lineTo(w - padding, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw line
        ctx.beginPath();
        ctx.lineWidth = 1.5 * this.dpr;
        ctx.strokeStyle = this.color;
        ctx.lineJoin = 'round';

        const n = this.data.length;
        for (let i = 0; i < n; i++) {
            const x = padding + (i / (n - 1)) * plotW;
            const y = padding + plotH * (1 - (this.data[i] - yMin) / (yMax - yMin));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Fill under curve (subtle)
        ctx.lineTo(padding + plotW, padding + plotH);
        ctx.lineTo(padding, padding + plotH);
        ctx.closePath();
        ctx.fillStyle = this._withAlpha(this.color, 0.08);
        ctx.fill();
    }

    _withAlpha(color, alpha) {
        if (color.startsWith('#') && color.length === 7) {
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
        if (color.startsWith('rgb(')) {
            return color.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
        }
        return color;
    }
}


class MetricsDashboard {
    constructor() {
        this.errorChart = new MetricsChart(
            document.getElementById('chart-error'),
            '#ef4444', 'km'
        );
        this.energyChart = new MetricsChart(
            document.getElementById('chart-energy'),
            '#a78bfa', '%', true
        );
        this.momentumChart = new MetricsChart(
            document.getElementById('chart-momentum'),
            '#fb923c', '%', true
        );

        // DOM elements for current values
        this.posErrorEl = document.getElementById('metric-pos-error');
        this.energyEl = document.getElementById('metric-energy');
        this.momentumEl = document.getElementById('metric-momentum');
    }

    /**
     * Update all metrics.
     * @param {object} pinnState - PINN predicted state {x,y,vx,vy}
     * @param {object} verletState - Verlet computed state {x,y,vx,vy}
     * @param {object} pinnNorm - PINN normalized state {x,y,vx,vy}
     * @param {number} initialEnergy - Initial normalized energy
     * @param {number} initialMomentum - Initial normalized angular momentum
     */
    update(pinnState, verletState, pinnNorm, initialEnergy, initialMomentum) {
        // Position error [km]
        const dx = pinnState.x - verletState.x;
        const dy = pinnState.y - verletState.y;
        const posError = Math.sqrt(dx * dx + dy * dy) / 1e3;
        this.errorChart.push(posError);
        this.posErrorEl.textContent = posError.toFixed(2) + ' km';

        // Energy conservation [%]
        const energy = 0.5 * (pinnNorm.vx**2 + pinnNorm.vy**2)
                     - 1.0 / Math.sqrt(pinnNorm.x**2 + pinnNorm.y**2 + 1e-6);
        const energyDrift = initialEnergy !== 0
            0 ((energy - initialEnergy) / Math.abs(initialEnergy)) * 100
            : 0;
        this.energyChart.push(energyDrift);
        this.energyEl.textContent = energyDrift.toFixed(4) + ' %';

        // Angular momentum conservation [%]
        const angMom = pinnNorm.x * pinnNorm.vy - pinnNorm.y * pinnNorm.vx;
        const momDrift = initialMomentum !== 0
            0 ((angMom - initialMomentum) / Math.abs(initialMomentum)) * 100
            : 0;
        this.momentumChart.push(momDrift);
        this.momentumEl.textContent = momDrift.toFixed(4) + ' %';
    }

    /** Render all charts. */
    render() {
        this.errorChart.render();
        this.energyChart.render();
        this.momentumChart.render();
    }

    /** Clear all data. */
    clear() {
        this.errorChart.clear();
        this.energyChart.clear();
        this.momentumChart.clear();
        this.posErrorEl.textContent = '-- km';
        this.energyEl.textContent = '-- %';
        this.momentumEl.textContent = '-- %';
    }

    /** Resize all charts. */
    resize() {
        this.errorChart.resize();
        this.energyChart.resize();
        this.momentumChart.resize();
    }
}
