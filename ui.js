/**
 * ui.js — Settings dashboard.
 *
 * Layout (top → bottom):
 *   Live stats · Rendering · Colors · Navigation · Bookmarks · Action bar
 */

import { drawGradientPreview, randomPalette } from './colors.js';
import { saveSettings, getBookmarks, addBookmark, removeBookmark } from './storage.js';

export class UI {
    constructor(container, settings, state, callbacks = {}) {
        this.container = container;
        this.settings = settings;
        this.state = state;
        this.callbacks = callbacks;

        this._visible = true;
        this._panel = null;
        this._fields = {};
        this._bookmarksList = null;
        this._gradientCanvas = null;
        this._colorStopsWrap = null;
        this._weightsWrap = null;
        this._floatBtn = null;
        this._helpEl = null;

        this._build();
    }

    // ─── Build ───────────────────────────────────────

    _build() {
        const panel = document.createElement('div');
        panel.className = 'dashboard';

        panel.appendChild(this._buildLive());
        panel.appendChild(this._divider('Rendering'));
        panel.appendChild(this._buildRendering());
        panel.appendChild(this._divider('Colors'));
        panel.appendChild(this._buildColors());
        panel.appendChild(this._divider('Navigation'));
        panel.appendChild(this._buildNavigation());
        panel.appendChild(this._divider('Bookmarks'));
        panel.appendChild(this._buildBookmarks());
        panel.appendChild(this._buildActionBar());

        const help = document.createElement('div');
        help.className = 'dash-help';
        panel.appendChild(help);
        this._helpEl = help;
        this._updateHelp();

        panel.addEventListener('wheel', (e) => e.stopPropagation());
        panel.addEventListener('mousedown', (e) => e.stopPropagation());

        this.container.appendChild(panel);
        this._panel = panel;

        // Floating toggle button for smartphone mode
        const floatBtn = document.createElement('button');
        floatBtn.className = 'float-toggle-btn hidden';
        floatBtn.textContent = '☰';
        floatBtn.title = 'Toggle settings';
        floatBtn.addEventListener('click', () => this.toggle());
        floatBtn.addEventListener('touchstart', (e) => e.stopPropagation());
        document.body.appendChild(floatBtn);
        this._floatBtn = floatBtn;

        this._applySmartphoneMode();
    }

    _divider(title) {
        const el = document.createElement('div');
        el.className = 'dash-divider';
        el.textContent = title;
        return el;
    }

    // ─── Live stats ──────────────────────────────────

    _buildLive() {
        const grid = document.createElement('div');
        grid.className = 'dash-stats';

        for (const [key, label] of [
            ['centerX', 'Center X'],
            ['centerY', 'Center Y'],
            ['zoom',    'Zoom'],
            ['iters',   'Iterations'],
            ['fps',     'FPS'],
            ['res',     'Resolution'],
        ]) {
            const lbl = document.createElement('span');
            lbl.className = 'stat-label';
            lbl.textContent = label;
            const val = document.createElement('span');
            val.className = 'stat-value';
            grid.appendChild(lbl);
            grid.appendChild(val);
            this._fields[key] = val;
        }

        return grid;
    }

    // ─── Rendering ───────────────────────────────────

    _buildRendering() {
        const wrap = document.createElement('div');

        // Segmented mode toggle
        const modeRow = document.createElement('div');
        modeRow.className = 'dash-row';
        const modeLbl = document.createElement('span');
        modeLbl.className = 'dash-label';
        modeLbl.textContent = 'Max iter mode';

        const seg = document.createElement('div');
        seg.className = 'seg-control';
        const mkBtn = (label, value) => {
            const btn = document.createElement('button');
            btn.className = 'seg-btn' + (this.state.maxiterMode === value ? ' active' : '');
            btn.textContent = label;
            btn.addEventListener('click', () => {
                this.state.maxiterMode = value;
                this.state.dirty = true;
                seg.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
            this._fields['maxiterMode_' + value] = btn;
            return btn;
        };
        seg.appendChild(mkBtn('Dynamic', 'Dynamic'));
        seg.appendChild(mkBtn('Fixed', 'Fixed'));
        modeRow.appendChild(modeLbl);
        modeRow.appendChild(seg);
        wrap.appendChild(modeRow);

        this._addSlider(wrap, 'Max iterations', 'baseMaxIter',
            10, 10000, 1, true,
            () => this.state.baseMaxIter,
            (v) => { this.state.baseMaxIter = v; this.state.dirty = true; }
        );

        return wrap;
    }

    // ─── Colors ──────────────────────────────────────

    _buildColors() {
        const wrap = document.createElement('div');

        this._addSlider(wrap, 'Color period', 'colorPeriod',
            8, 4096, 1, true,
            () => this.settings.colorPeriod,
            (v) => { this.settings.colorPeriod = v; this.state.dirty = true; }
        );

        // Gradient preview
        const gradCanvas = document.createElement('canvas');
        gradCanvas.className = 'gradient-preview';
        gradCanvas.width = 300;
        gradCanvas.height = 16;
        wrap.appendChild(gradCanvas);
        this._gradientCanvas = gradCanvas;
        drawGradientPreview(gradCanvas, this.settings.colors, this.settings.weights, this.settings.gradientFunction);

        // Color stops + add/remove on same row
        const stopsRow = document.createElement('div');
        stopsRow.className = 'dash-row dash-row--wrap';

        this._colorStopsWrap = document.createElement('div');
        this._colorStopsWrap.className = 'color-stops';
        this._rebuildColorStops();

        const addBtn = document.createElement('button');
        addBtn.className = 'dash-btn dash-btn--icon';
        addBtn.title = 'Add color stop';
        addBtn.textContent = '+';
        addBtn.addEventListener('click', () => {
            const last = this.settings.colors[this.settings.colors.length - 1];
            this.settings.colors.push({ ...last });
            this.settings.weights.push(1.0);
            this._onStopsChanged();
        });

        const removeBtn = document.createElement('button');
        removeBtn.className = 'dash-btn dash-btn--icon dash-btn--danger';
        removeBtn.title = 'Remove last stop';
        removeBtn.textContent = '−';
        removeBtn.addEventListener('click', () => {
            if (this.settings.colors.length <= 2) return;
            this.settings.colors.pop();
            this.settings.weights.pop();
            this._onStopsChanged();
        });

        const randomBtn = document.createElement('button');
        randomBtn.className = 'dash-btn';
        randomBtn.textContent = '🎲 Randomize';
        randomBtn.addEventListener('click', () => {
            const palette = randomPalette();
            this.settings.colors      = palette.colors;
            this.settings.weights     = palette.weights;
            this.settings.colorPeriod = palette.colorPeriod;
            this._fields['colorPeriod_slider'] && (this._fields['colorPeriod_slider'].value = palette.colorPeriod);
            this._fields['colorPeriod_num']    && (this._fields['colorPeriod_num'].value    = palette.colorPeriod);
            this._onStopsChanged();
        });

        stopsRow.appendChild(this._colorStopsWrap);
        stopsRow.appendChild(addBtn);
        stopsRow.appendChild(removeBtn);
        wrap.appendChild(stopsRow);
        wrap.appendChild(randomBtn);

        // Segment weights
        const wLbl = document.createElement('div');
        wLbl.className = 'dash-sublabel';
        wLbl.textContent = 'Segment weights';
        wrap.appendChild(wLbl);

        this._weightsWrap = document.createElement('div');
        this._weightsWrap.className = 'weights-row';
        wrap.appendChild(this._weightsWrap);
        this._rebuildWeights();

        return wrap;
    }

    _rebuildColorStops() {
        const wrap = this._colorStopsWrap;
        wrap.innerHTML = '';
        this.settings.colors.forEach((c, i) => {
            const hex = '#' +
                c.r.toString(16).padStart(2, '0') +
                c.g.toString(16).padStart(2, '0') +
                c.b.toString(16).padStart(2, '0');
            const picker = document.createElement('input');
            picker.type = 'color';
            picker.className = 'color-swatch';
            picker.value = hex;
            picker.title = `Stop ${i + 1}`;
            picker.addEventListener('input', () => {
                const h = picker.value;
                this.settings.colors[i] = {
                    r: parseInt(h.slice(1, 3), 16),
                    g: parseInt(h.slice(3, 5), 16),
                    b: parseInt(h.slice(5, 7), 16),
                    a: 255,
                };
                this._onGradientChange();
            });
            wrap.appendChild(picker);
        });
    }

    _rebuildWeights() {
        const wrap = this._weightsWrap;
        wrap.innerHTML = '';
        this.settings.weights.forEach((w, i) => {
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.className = 'weight-input';
            inp.min = 0.1; inp.max = 20; inp.step = 0.1;
            inp.value = w;
            inp.title = `Segment ${i + 1}`;
            inp.addEventListener('change', () => {
                const v = Math.max(0.1, parseFloat(inp.value) || 1);
                inp.value = v;
                this.settings.weights[i] = v;
                this._onGradientChange();
            });
            wrap.appendChild(inp);
        });
    }

    // Called when a color value or weight changes — only redraw the preview.
    // Do NOT rebuild the stop pickers here: destroying the DOM node closes the
    // native color picker dialog immediately.
    _onGradientChange() {
        this.settings._colorVersion = (this.settings._colorVersion || 0) + 1;
        this.state.dirty = true;
        drawGradientPreview(this._gradientCanvas, this.settings.colors, this.settings.weights, this.settings.gradientFunction);
    }

    // Called when the number of stops changes (add/remove/randomize).
    // Safe to rebuild the full DOM here because no picker is open.
    _onStopsChanged() {
        this._onGradientChange();
        this._rebuildColorStops();
        this._rebuildWeights();
    }

    // ─── Navigation ──────────────────────────────────

    _buildNavigation() {
        const wrap = document.createElement('div');

        this._addSlider(wrap, 'Scroll zoom speed', 'zoomSpeed',
            0.50, 0.99, 0.01, false,
            () => this.settings.zoomSpeed,
            (v) => { this.settings.zoomSpeed = v; }
        );
        this._addSlider(wrap, 'Key zoom speed', 'keyZoomSpeed',
            0.50, 0.99, 0.01, false,
            () => this.settings.keyZoomSpeed,
            (v) => { this.settings.keyZoomSpeed = v; }
        );
        this._addSlider(wrap, 'Pan speed', 'panSpeed',
            0.01, 0.40, 0.01, false,
            () => this.settings.panSpeed,
            (v) => { this.settings.panSpeed = v; }
        );

        // Smartphone mode toggle
        const row = document.createElement('div');
        row.className = 'dash-row';
        const lbl = document.createElement('label');
        lbl.className = 'dash-label';
        lbl.textContent = 'Smartphone mode';
        const seg = document.createElement('div');
        seg.className = 'seg-control';
        const mkBtn = (label, value) => {
            const btn = document.createElement('button');
            btn.className = 'seg-btn' + (this.state.smartphoneMode === value ? ' active' : '');
            btn.textContent = label;
            btn.addEventListener('click', () => {
                this.state.smartphoneMode = value;
                seg.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this._applySmartphoneMode();
            });
            this._fields['smartphone_' + label] = btn;
            return btn;
        };
        seg.appendChild(mkBtn('Off', false));
        seg.appendChild(mkBtn('On', true));
        row.appendChild(lbl);
        row.appendChild(seg);
        wrap.appendChild(row);

        return wrap;
    }

    // ─── Bookmarks ───────────────────────────────────

    _buildBookmarks() {
        const wrap = document.createElement('div');

        const saveRow = document.createElement('div');
        saveRow.className = 'dash-row';

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.className = 'dash-text-input';
        nameInput.placeholder = 'Name…';
        nameInput.maxLength = 60;

        const saveBtn = document.createElement('button');
        saveBtn.className = 'dash-btn';
        saveBtn.textContent = 'Save current';
        saveBtn.addEventListener('click', () => {
            const name = nameInput.value.trim() || `Point ${getBookmarks().length + 1}`;
            addBookmark({
                name,
                x: this.state.centerBF_X
                    ? this.state.centerBF_X.toDecimalString(40)
                    : String(this.state.centerX),
                y: this.state.centerBF_Y
                    ? this.state.centerBF_Y.toDecimalString(40)
                    : String(this.state.centerY),
                viewport: this.state.viewportSizeY,
            });
            nameInput.value = '';
            this._refreshBookmarksList();
        });

        saveRow.appendChild(nameInput);
        saveRow.appendChild(saveBtn);
        wrap.appendChild(saveRow);

        this._bookmarksList = document.createElement('div');
        this._bookmarksList.className = 'bookmarks-list';
        wrap.appendChild(this._bookmarksList);
        this._refreshBookmarksList();

        return wrap;
    }

    _refreshBookmarksList() {
        const list = this._bookmarksList;
        list.innerHTML = '';
        const bookmarks = getBookmarks();

        if (bookmarks.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'bookmarks-empty';
            empty.textContent = 'No saved points yet.';
            list.appendChild(empty);
            return;
        }

        bookmarks.forEach((bm, idx) => {
            const row = document.createElement('div');
            row.className = 'bookmark-row';

            const goBtn = document.createElement('button');
            goBtn.className = 'dash-btn dash-btn--go';
            goBtn.textContent = '▶';
            goBtn.title = 'Navigate here';
            goBtn.addEventListener('click', () => {
                if (this.callbacks.navigateTo) this.callbacks.navigateTo(bm.x, bm.y, bm.viewport);
            });

            const info = document.createElement('div');
            info.className = 'bookmark-info';
            const nameLine = document.createElement('div');
            nameLine.className = 'bookmark-name';
            nameLine.textContent = bm.name;
            const zoomLine = document.createElement('div');
            zoomLine.className = 'bookmark-zoom';
            zoomLine.textContent = `zoom 10^${(-Math.log10(bm.viewport)).toFixed(1)}`;
            info.appendChild(nameLine);
            info.appendChild(zoomLine);

            const delBtn = document.createElement('button');
            delBtn.className = 'dash-btn dash-btn--danger dash-btn--icon';
            delBtn.textContent = '×';
            delBtn.title = 'Delete';
            delBtn.addEventListener('click', () => { removeBookmark(idx); this._refreshBookmarksList(); });

            row.appendChild(goBtn);
            row.appendChild(info);
            row.appendChild(delBtn);
            list.appendChild(row);
        });
    }

    // ─── Action bar ──────────────────────────────────

    _buildActionBar() {
        const bar = document.createElement('div');
        bar.className = 'action-bar';

        const screenshotBtn = document.createElement('button');
        screenshotBtn.className = 'dash-btn action-btn';
        screenshotBtn.textContent = '📷 Screenshot';
        screenshotBtn.addEventListener('click', () => {
            if (this.callbacks.takeScreenshot) this.callbacks.takeScreenshot();
        });

        const saveBtn = document.createElement('button');
        saveBtn.className = 'dash-btn action-btn';
        saveBtn.textContent = 'Save settings';
        saveBtn.addEventListener('click', () => {
            saveSettings(this.settings, this.state);
            saveBtn.textContent = 'Saved ✓';
            setTimeout(() => { saveBtn.textContent = 'Save settings'; }, 1500);
        });

        const resetBtn = document.createElement('button');
        resetBtn.className = 'dash-btn action-btn dash-btn--danger';
        resetBtn.textContent = 'Reset defaults';
        resetBtn.addEventListener('click', () => {
            const def = this.callbacks.defaultSettings;
            if (!def) return;
            this.settings.colors              = def.colors.map(c => ({ ...c }));
            this.settings.weights             = [...def.weights];
            this.settings.colorPeriod         = def.colorPeriod;
            this.settings.zoomSpeed           = def.zoomSpeed;
            this.settings.panSpeed            = def.panSpeed;
            this.settings.keyZoomSpeed        = def.keyZoomSpeed;
            this.settings.maxIterAdjustFactor = def.maxIterAdjustFactor;
            this.settings._colorVersion       = (this.settings._colorVersion || 0) + 1;
            this.state.baseMaxIter            = def.initialMaxIter;
            this.state.maxiterMode            = def.maxiterMode;
            this.state.dirty                  = true;
            this._syncInputs();
            this._onStopsChanged();
        });

        bar.appendChild(screenshotBtn);
        bar.appendChild(saveBtn);
        bar.appendChild(resetBtn);
        return bar;
    }

    // ─── Slider helper ───────────────────────────────

    _addSlider(parent, label, key, min, max, step, integer, getter, setter) {
        const row = document.createElement('div');
        row.className = 'dash-row';

        const lbl = document.createElement('label');
        lbl.className = 'dash-label';
        lbl.textContent = label;

        const group = document.createElement('div');
        group.className = 'dash-slider-group';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'dash-slider';
        slider.min = min; slider.max = max; slider.step = step;
        slider.value = getter();

        const num = document.createElement('input');
        num.type = 'number';
        num.className = 'dash-number';
        num.min = min; num.max = max; num.step = step;
        num.value = getter();

        const sync = (raw) => {
            const parsed = integer ? parseInt(raw, 10) : parseFloat(raw);
            const v = Math.max(min, Math.min(max, isNaN(parsed) ? getter() : parsed));
            slider.value = v; num.value = v;
            setter(v);
        };

        slider.addEventListener('input', () => sync(slider.value));
        num.addEventListener('change', () => sync(num.value));

        group.appendChild(slider);
        group.appendChild(num);
        row.appendChild(lbl);
        row.appendChild(group);
        parent.appendChild(row);

        this._fields[key + '_slider'] = slider;
        this._fields[key + '_num'] = num;
    }

    // ─── Public API ──────────────────────────────────

    toggle() {
        this._visible = !this._visible;
        this._panel.classList.toggle('hidden', !this._visible);
        if (this._floatBtn) this._floatBtn.classList.toggle('panel-open', this._visible);
        if (this._visible) { this._syncInputs(); this._refreshBookmarksList(); }
    }

    _applySmartphoneMode() {
        const on = this.state.smartphoneMode;
        if (this._floatBtn) this._floatBtn.classList.toggle('hidden', !on);
        this._updateHelp();
    }

    _updateHelp() {
        if (!this._helpEl) return;
        if (this.state.smartphoneMode) {
            this._helpEl.innerHTML = 'Drag=Pan &nbsp;·&nbsp; Pinch=Zoom &nbsp;·&nbsp; ☰=Toggle panel';
        } else {
            this._helpEl.innerHTML = 'Drag=Pan &nbsp;·&nbsp; Scroll/W/S=Zoom &nbsp;·&nbsp; Arrows=Pan &nbsp;·&nbsp; I=Toggle panel &nbsp;·&nbsp; F11=Fullscreen';
        }
    }

    _syncInputs() {
        const set = (k, v) => {
            if (this._fields[k + '_slider']) this._fields[k + '_slider'].value = v;
            if (this._fields[k + '_num'])    this._fields[k + '_num'].value = v;
        };
        set('baseMaxIter', this.state.baseMaxIter);
        set('colorPeriod', this.settings.colorPeriod);
        set('zoomSpeed', this.settings.zoomSpeed);
        set('keyZoomSpeed', this.settings.keyZoomSpeed);
        set('panSpeed', this.settings.panSpeed);

        // Segmented mode buttons
        ['Dynamic', 'Fixed'].forEach(v => {
            const btn = this._fields['maxiterMode_' + v];
            if (btn) btn.classList.toggle('active', this.state.maxiterMode === v);
        });

        // Smartphone mode buttons
        [['Off', false], ['On', true]].forEach(([label, value]) => {
            const btn = this._fields['smartphone_' + label];
            if (btn) btn.classList.toggle('active', this.state.smartphoneMode === value);
        });
    }

    update(state) {
        if (!this._visible) return;
        this._fields.centerX.textContent = state.centerBF_X
            ? state.centerBF_X.toDecimalString(40) : state.centerX.toFixed(17);
        this._fields.centerY.textContent = state.centerBF_Y
            ? state.centerBF_Y.toDecimalString(40) : state.centerY.toFixed(17);
        this._fields.zoom.textContent = `10^${(-Math.log10(state.viewportSizeY)).toFixed(1)}`;
        this._fields.iters.textContent = String(state.maxIter);
        this._fields.fps.textContent = String(state.fps);
        this._fields.res.textContent = `${state.width} × ${state.height}`;
    }

    destroy() {
        if (this._panel && this._panel.parentNode) this._panel.parentNode.removeChild(this._panel);
        if (this._floatBtn && this._floatBtn.parentNode) this._floatBtn.parentNode.removeChild(this._floatBtn);
    }
}
