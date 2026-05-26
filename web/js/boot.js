class BootSequence {
    constructor() {
        this.overlay = document.getElementById('boot-overlay');
        this.logEl = document.getElementById('boot-log');
        this.progressBar = document.getElementById('boot-progress-bar');
        this.app = document.getElementById('app');

        this.messages = [
            { text: 'KeplerLab evidence console', cls: 'accent', delay: 120 },
            { text: 'Two-body traces indexed', cls: 'info', delay: 120 },
            { text: 'Autopsy stages prepared', cls: 'info', delay: 120 },
            { text: 'Canvas diagnostics armed', cls: 'success', delay: 140 },
            { text: 'Ready', cls: 'accent', delay: 160 },
        ];
    }

    async run(onComplete) {
        for (let i = 0; i < this.messages.length; i++) {
            const msg = this.messages[i];
            await this._delay(msg.delay);
            this._addLine(msg.text, msg.cls);
            this._setProgress(((i + 1) / this.messages.length) * 100);
        }

        await this._delay(600);
        this.overlay.classList.add('fade-out');
        await this._delay(300);
        this.app.classList.remove('hidden');
        await this._delay(500);
        this.overlay.style.display = 'none';

        if (onComplete) onComplete();
    }

    addMessage(text, cls = 'info') {
        this._addLine(text, cls);
    }

    _addLine(text, cls) {
        const line = document.createElement('div');
        line.className = `log-line ${cls}`;
        line.textContent = text;
        this.logEl.appendChild(line);
        this.logEl.scrollTop = this.logEl.scrollHeight;
    }

    _setProgress(percent) {
        this.progressBar.style.width = `${Math.min(percent, 100)}%`;
    }

    _delay(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }
}
