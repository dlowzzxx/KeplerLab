class KeplerBoard {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.trace = null;
        this.index = 0;
        this.showError = true;
        this.showCollocation = false;
        this.dpr = window.devicePixelRatio || 1;
        this.resize();
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        const cssWidth = Math.max(320, rect.width || this.canvas.clientWidth || 900);
        const cssHeight = Math.max(320, rect.height || this.canvas.clientHeight || 620);
        this.canvas.width = Math.round(cssWidth * this.dpr);
        this.canvas.height = Math.round(cssHeight * this.dpr);
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.render();
    }

    setTrace(trace) {
        this.trace = trace;
        this.index = 0;
        this._computeScale();
        this.render();
    }

    setIndex(index) {
        if (!this.trace) return;
        this.index = Math.max(0, Math.min(this.trace.series.length - 1, index));
        this.render();
    }

    setOverlays({ showError, showCollocation }) {
        this.showError = showError;
        this.showCollocation = showCollocation;
        this.render();
    }

    _computeScale() {
        if (!this.trace) return;
        let maxAbs = 1;
        for (const point of this.trace.series) {
            maxAbs = Math.max(
                maxAbs,
                Math.abs(point.gt[0]),
                Math.abs(point.gt[1]),
            );
            if (point.pred) {
                maxAbs = Math.max(maxAbs, Math.abs(point.pred[0]), Math.abs(point.pred[1]));
            }
        }
        this.maxKm = maxAbs * 1.18;
    }

    _toPx(pos) {
        const margin = 48 * this.dpr;
        const size = Math.min(this.width, this.height) - margin * 2;
        const scale = size / (2 * this.maxKm);
        return {
            x: this.width / 2 + pos[0] * scale,
            y: this.height / 2 - pos[1] * scale,
            scale,
        };
    }

    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);
        ctx.fillStyle = '#09090b';
        ctx.fillRect(0, 0, this.width, this.height);

        this._drawGrid();
        if (!this.trace) {
            this._drawEmpty();
            return;
        }

        this._drawPlotFrame();
        this._drawOrbitLabels();
        this._drawEarth();
        this._drawResidualHeatStrip();
        this._drawPath('gt', '#f97316', 0.72, 2.2);
        if (this._hasPrediction()) {
            this._drawPath('pred', '#34d399', 0.9, 1.6, [8, 6]);
        }
        if (this.showError) this._drawErrorVectors();
        this._drawPhaseLens();
        this._drawCurrentState();
        if (this.showCollocation) this._drawCollocationStrip();
        this._drawLegend();
    }

    _drawEmpty() {
        const ctx = this.ctx;
        ctx.fillStyle = '#64706d';
        ctx.font = `${13 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText('Waiting for evidence trace...', 28 * this.dpr, 42 * this.dpr);
    }

    _drawGrid() {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = 'rgba(100,112,107,0.18)';
        ctx.lineWidth = this.dpr;
        const step = 64 * this.dpr;
        for (let x = 0; x <= this.width; x += step) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.height);
            ctx.stroke();
        }
        for (let y = 0; y <= this.height; y += step) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.width, y);
            ctx.stroke();
        }
        ctx.strokeStyle = 'rgba(154,167,162,0.26)';
        ctx.beginPath();
        ctx.moveTo(this.width / 2, 0);
        ctx.lineTo(this.width / 2, this.height);
        ctx.moveTo(0, this.height / 2);
        ctx.lineTo(this.width, this.height / 2);
        ctx.stroke();
        ctx.restore();
    }

    _drawPlotFrame() {
        const ctx = this.ctx;
        const pad = 16 * this.dpr;
        const tick = 7 * this.dpr;
        ctx.save();
        ctx.strokeStyle = 'rgba(174,181,170,0.32)';
        ctx.lineWidth = this.dpr;
        ctx.strokeRect(pad, pad, this.width - pad * 2, this.height - pad * 2);

        for (let i = 1; i < 8; i++) {
            const x = pad + (i / 8) * (this.width - pad * 2);
            ctx.beginPath();
            ctx.moveTo(x, pad);
            ctx.lineTo(x, pad + tick);
            ctx.moveTo(x, this.height - pad);
            ctx.lineTo(x, this.height - pad - tick);
            ctx.stroke();
        }
        for (let i = 1; i < 5; i++) {
            const y = pad + (i / 5) * (this.height - pad * 2);
            ctx.beginPath();
            ctx.moveTo(pad, y);
            ctx.lineTo(pad + tick, y);
            ctx.moveTo(this.width - pad, y);
            ctx.lineTo(this.width - pad - tick, y);
            ctx.stroke();
        }

        ctx.fillStyle = '#737b70';
        ctx.font = `${9 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText('km phase plane / Earth-centered inertial slice', pad + 8 * this.dpr, pad + 15 * this.dpr);
        ctx.restore();
    }

    _drawEarth() {
        const ctx = this.ctx;
        const center = this._toPx([0, 0]);
        const r = 6371 * center.scale;
        ctx.save();
        ctx.fillStyle = '#14212a';
        ctx.strokeStyle = '#67e8f9';
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(center.x, center.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 0.38;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#71717a';
        ctx.font = `${10 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText('Earth', center.x + r + 8 * this.dpr, center.y + 16 * this.dpr);
        ctx.restore();
    }

    _drawResidualHeatStrip() {
        if (!this._hasPrediction()) return;
        const ctx = this.ctx;
        const w = this.width - 96 * this.dpr;
        const h = 6 * this.dpr;
        const x0 = 48 * this.dpr;
        const y0 = this.height - 24 * this.dpr;
        
        ctx.save();
        ctx.fillStyle = 'rgba(39,39,42,0.8)';
        ctx.fillRect(x0, y0, w, h);
        
        let maxErr = 0;
        for (const p of this.trace.series) {
            if (p.err) maxErr = Math.max(maxErr, p.err);
        }
        if (maxErr < 1e-6) maxErr = 1;

        for (let i = 0; i < this.trace.series.length; i++) {
            const p = this.trace.series[i];
            if (!p.err) continue;
            const x = x0 + (i / (this.trace.series.length - 1)) * w;
            const intensity = Math.min(1, p.err / (maxErr * 0.5));
            ctx.fillStyle = `rgba(239, 68, 68, ${intensity})`;
            ctx.fillRect(x, y0, 2 * this.dpr, h);
        }

        ctx.fillStyle = '#71717a';
        ctx.font = `${9 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText('residual error phase distribution', x0, y0 - 6 * this.dpr);
        ctx.restore();
    }

    _drawPhaseLens() {
        if (!this.trace || this.trace.orbit.eccentricity < 0.3) return;
        
        const ctx = this.ctx;
        let periPoint = this.trace.series[0];
        let minR = Infinity;
        for (const p of this.trace.series) {
            const r = Math.hypot(p.gt[0], p.gt[1]);
            if (r < minR) { minR = r; periPoint = p; }
        }

        const insetX = 34 * this.dpr;
        const insetY = this.height - 186 * this.dpr;
        const insetW = Math.min(230 * this.dpr, this.width * 0.32);
        const insetH = 124 * this.dpr;
        const zoom = 4;

        ctx.save();
        ctx.beginPath();
        ctx.rect(insetX, insetY, insetW, insetH);
        ctx.clip();

        ctx.fillStyle = '#121214';
        ctx.fillRect(insetX, insetY, insetW, insetH);
        ctx.strokeStyle = 'rgba(69,73,63,0.55)';
        ctx.lineWidth = this.dpr;
        for (let i = 1; i < 5; i++) {
            const x = insetX + (i / 5) * insetW;
            ctx.beginPath();
            ctx.moveTo(x, insetY);
            ctx.lineTo(x, insetY + insetH);
            ctx.stroke();
        }

        const drawMagnifiedPath = (key, color, width, dash = null) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = width * this.dpr;
            if (dash) ctx.setLineDash(dash.map(v => v * this.dpr));
            ctx.beginPath();
            let started = false;
            for (const p of this.trace.series) {
                if (!p[key]) continue;
                const px = this._toPx(p[key]);
                const periPx = this._toPx(periPoint.gt);
                const dx = px.x - periPx.x;
                const dy = px.y - periPx.y;
                const mx = insetX + insetW * 0.56 + dx * zoom;
                const my = insetY + insetH * 0.55 + dy * zoom;
                if (!started) { ctx.moveTo(mx, my); started = true; }
                else ctx.lineTo(mx, my);
            }
            if (started) ctx.stroke();
        };

        drawMagnifiedPath('gt', '#f97316', 2.2);
        if (this._hasPrediction()) {
            drawMagnifiedPath('pred', '#34d399', 1.6, [8, 6]);
            
            // Error vectors in lens
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = this.dpr;
            ctx.setLineDash([]);
            for (const p of this.trace.series) {
                if (!p.pred) continue;
                const gtPx = this._toPx(p.gt);
                const predPx = this._toPx(p.pred);
                const periPx = this._toPx(periPoint.gt);
                ctx.beginPath();
                ctx.moveTo(insetX + insetW * 0.56 + (gtPx.x - periPx.x) * zoom, insetY + insetH * 0.55 + (gtPx.y - periPx.y) * zoom);
                ctx.lineTo(insetX + insetW * 0.56 + (predPx.x - periPx.x) * zoom, insetY + insetH * 0.55 + (predPx.y - periPx.y) * zoom);
                ctx.stroke();
            }
        }

        ctx.restore();
        
        ctx.save();
        ctx.strokeStyle = '#3f3f46';
        ctx.lineWidth = 2 * this.dpr;
        ctx.strokeRect(insetX, insetY, insetW, insetH);
        ctx.fillStyle = '#71717a';
        ctx.font = `${9 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText(`periapsis lens ${zoom}x`, insetX + 8 * this.dpr, insetY + 14 * this.dpr);
        ctx.restore();
    }

    _drawPath(key, color, alpha, width, dash = null) {
        const ctx = this.ctx;
        const series = this.trace.series;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = width * this.dpr;
        if (dash) ctx.setLineDash(dash.map((v) => v * this.dpr));
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < series.length; i++) {
            if (!series[i][key]) continue;
            const p = this._toPx(series[i][key]);
            if (!started) {
                ctx.moveTo(p.x, p.y);
                started = true;
            }
            else ctx.lineTo(p.x, p.y);
        }
        if (started) ctx.stroke();
        ctx.restore();
    }

    _drawErrorVectors() {
        const ctx = this.ctx;
        const series = this.trace.series;
        const stride = Math.max(18, Math.floor(series.length / 34));
        ctx.save();
        ctx.strokeStyle = '#ef4444';
        ctx.globalAlpha = 0.48;
        ctx.lineWidth = this.dpr;
        for (let i = 0; i < series.length; i += stride) {
            if (!series[i].pred) continue;
            const a = this._toPx(series[i].gt);
            const b = this._toPx(series[i].pred);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        }
        ctx.restore();
    }

    _drawCurrentState() {
        const ctx = this.ctx;
        const point = this.trace.series[this.index];
        const gt = this._toPx(point.gt);

        ctx.save();
        this._dot(gt.x, gt.y, '#f97316', 4.5);
        if (point.pred) {
            const pred = this._toPx(point.pred);
            ctx.lineWidth = 1.2 * this.dpr;
            ctx.strokeStyle = '#ef4444';
            ctx.beginPath();
            ctx.moveTo(gt.x, gt.y);
            ctx.lineTo(pred.x, pred.y);
            ctx.stroke();
            this._dot(pred.x, pred.y, '#34d399', 4.5);
        }

        ctx.fillStyle = '#edf2ef';
        ctx.font = `${11 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText(`orbit ${point.orbit.toFixed(3)}`, 18 * this.dpr, 28 * this.dpr);
        const errorText = Number.isFinite(point.err)
            ? `error ${point.err.toFixed(2)} km`
            : 'prediction pending';
        ctx.fillText(errorText, 18 * this.dpr, 46 * this.dpr);
        ctx.restore();
    }

    _dot(x, y, color, r) {
        const ctx = this.ctx;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, r * this.dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#050606';
        ctx.lineWidth = 1.5 * this.dpr;
        ctx.stroke();
    }

    _drawOrbitLabels() {
        const ctx = this.ctx;
        let periPoint = this.trace.series[0];
        let apoPoint = this.trace.series[0];
        let minR = Infinity;
        let maxR = -Infinity;
        for (const point of this.trace.series) {
            const r = Math.hypot(point.gt[0], point.gt[1]);
            if (r < minR) {
                minR = r;
                periPoint = point;
            }
            if (r > maxR) {
                maxR = r;
                apoPoint = point;
            }
        }
        const peri = this._toPx(periPoint.gt);
        const apo = this._toPx(apoPoint.gt);
        ctx.save();
        ctx.fillStyle = '#71717a';
        ctx.font = `${10 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText('periapsis', peri.x + 8 * this.dpr, peri.y - 8 * this.dpr);
        ctx.fillText('apoapsis', apo.x + 8 * this.dpr, apo.y - 8 * this.dpr);
        ctx.restore();
    }

    _drawCollocationStrip() {
        const ctx = this.ctx;
        const collocation = this.trace.collocation;
        if (!collocation) return;

        const x0 = 18 * this.dpr;
        const y0 = this.height - 74 * this.dpr;
        const w = Math.min(this.width - 36 * this.dpr, 460 * this.dpr);
        const h = 44 * this.dpr;
        const maxOrbit = Math.max(...collocation.uniform, ...collocation.focused, 1);

        ctx.save();
        ctx.fillStyle = 'rgba(13,15,15,0.86)';
        ctx.strokeStyle = 'rgba(58,70,70,0.9)';
        ctx.fillRect(x0, y0, w, h);
        ctx.strokeRect(x0, y0, w, h);
        ctx.font = `${9 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillStyle = '#9ba7a2';
        ctx.fillText('collocation: uniform vs periapsis-focused', x0 + 8 * this.dpr, y0 + 14 * this.dpr);

        for (const value of collocation.uniform) {
            const x = x0 + (value / maxOrbit) * w;
            ctx.fillStyle = 'rgba(103,232,249,0.20)';
            ctx.fillRect(x, y0 + 23 * this.dpr, this.dpr, 7 * this.dpr);
        }
        for (const value of collocation.focused) {
            const x = x0 + (value / maxOrbit) * w;
            ctx.fillStyle = 'rgba(52,211,153,0.45)';
            ctx.fillRect(x, y0 + 32 * this.dpr, this.dpr, 7 * this.dpr);
        }
        ctx.restore();
    }

    _drawLegend() {
        const ctx = this.ctx;
        const x = this.width - 196 * this.dpr;
        const y = 18 * this.dpr;
        ctx.save();
        ctx.fillStyle = 'rgba(13,15,15,0.82)';
        ctx.strokeStyle = 'rgba(58,70,70,0.9)';
        ctx.fillRect(x, y, 174 * this.dpr, 72 * this.dpr);
        ctx.strokeRect(x, y, 174 * this.dpr, 72 * this.dpr);
        this._legendLine(x + 12 * this.dpr, y + 20 * this.dpr, '#f97316', 'Velocity Verlet');
        if (this._hasPrediction()) {
            this._legendLine(x + 12 * this.dpr, y + 42 * this.dpr, '#34d399', 'PINN / reconstruction');
            this._legendLine(x + 12 * this.dpr, y + 64 * this.dpr, '#ef4444', 'error vector');
        } else {
            this._legendLine(x + 12 * this.dpr, y + 42 * this.dpr, '#34d399', 'model pending');
        }
        ctx.restore();
    }

    _legendLine(x, y, color, label) {
        const ctx = this.ctx;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 * this.dpr;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 22 * this.dpr, y);
        ctx.stroke();
        ctx.fillStyle = '#9ba7a2';
        ctx.font = `${9 * this.dpr}px JetBrains Mono, monospace`;
        ctx.fillText(label, x + 30 * this.dpr, y + 3 * this.dpr);
    }

    _hasPrediction() {
        return Boolean(this.trace && this.trace.series.some((point) => point.pred));
    }
}
