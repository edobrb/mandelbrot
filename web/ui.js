/**
 * ui.js — Info overlay, gradient preview bar, on-screen help.
 */

import { drawGradientPreview } from './colors.js';

export class UI {
    constructor(container, settings) {
        this.container = container;
        this.settings = settings;

        this._visible = false;
        this._overlay = null;
        this._gradientCanvas = null;
        this._fields = {};

        this._build();
    }

    _build() {
        // Overlay container
        const overlay = document.createElement('div');
        overlay.id = 'info-overlay';
        overlay.className = 'info-overlay hidden';

        // Info fields
        const fields = [
            ['centerX',  'Center X'],
            ['centerY',  'Center Y'],
            ['viewport', 'Viewport Y'],
            ['maxIter',  'Max Iter'],
            ['fps',      'FPS'],
            ['res',      'Resolution'],
        ];

        const table = document.createElement('div');
        table.className = 'info-table';

        for (const [key, label] of fields) {
            const row = document.createElement('div');
            row.className = 'info-row';

            const lbl = document.createElement('span');
            lbl.className = 'info-label';
            lbl.textContent = label + ':';

            const val = document.createElement('span');
            val.className = 'info-value';
            val.id = `info-${key}`;

            row.appendChild(lbl);
            row.appendChild(val);
            table.appendChild(row);
            this._fields[key] = val;
        }

        overlay.appendChild(table);

        // Gradient preview
        const gradLabel = document.createElement('div');
        gradLabel.className = 'info-grad-label';
        gradLabel.textContent = 'Color Gradient:';
        overlay.appendChild(gradLabel);

        const gradCanvas = document.createElement('canvas');
        gradCanvas.className = 'gradient-preview';
        gradCanvas.width = 300;
        gradCanvas.height = 20;
        overlay.appendChild(gradCanvas);
        this._gradientCanvas = gradCanvas;

        // Controls help
        const help = document.createElement('div');
        help.className = 'info-help';
        help.innerHTML =
            '<b>Controls:</b> Drag=Pan · Scroll=Zoom · W/S=Zoom · Arrows=Pan · M/N=Iter · I=Info · F11=Fullscreen · F12=Screenshot';
        overlay.appendChild(help);

        this.container.appendChild(overlay);
        this._overlay = overlay;

        // Draw initial gradient
        drawGradientPreview(
            this._gradientCanvas,
            this.settings.colors,
            this.settings.weights,
            this.settings.gradientFunction,
        );
    }

    toggle() {
        this._visible = !this._visible;
        this._overlay.classList.toggle('hidden', !this._visible);
    }

    update(state) {
        if (!this._visible) return;

        this._fields.centerX.textContent = state.centerX.toFixed(17);
        this._fields.centerY.textContent = state.centerY.toFixed(17);
        this._fields.viewport.textContent = state.viewportSizeY.toExponential(6);
        this._fields.maxIter.textContent = String(state.maxIter);
        this._fields.fps.textContent = String(state.fps);
        this._fields.res.textContent = `${state.width} × ${state.height}`;
    }

    destroy() {
        if (this._overlay && this._overlay.parentNode) {
            this._overlay.parentNode.removeChild(this._overlay);
        }
    }
}
