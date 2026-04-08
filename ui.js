/**
 * ui.js — Settings dashboard.
 *
 * Layout (top → bottom):
 *   Live stats · Rendering · Colors · Navigation · Bookmarks · Action bar
 */

import { drawGradientPreview, randomPalette } from './colors.js';
import { saveSettings, getBookmarks, addBookmark, removeBookmark, encodeShareHash } from './storage.js';

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
        this._geCanvas = null;
        this._geRail = null;
        this._geEditor = null;
        this._geActiveIdx = -1;
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

        // Floating toggle button (always visible)
        const floatBtn = document.createElement('button');
        floatBtn.className = 'float-toggle-btn panel-open';
        floatBtn.textContent = '☰';
        floatBtn.title = 'Toggle settings';
        floatBtn.addEventListener('click', () => this.toggle());
        floatBtn.addEventListener('touchstart', (e) => e.stopPropagation());
        document.body.appendChild(floatBtn);
        this._floatBtn = floatBtn;
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

        // ── Gradient Editor ──
        const ge = document.createElement('div');
        ge.className = 'ge';

        // Toolbar: hint text + randomize button
        const toolbar = document.createElement('div');
        toolbar.className = 'ge-toolbar';
        const hint = document.createElement('span');
        hint.className = 'ge-hint';
        hint.textContent = 'Click gradient to add · click handle to edit';
        const randomBtn = document.createElement('button');
        randomBtn.className = 'dash-btn ge-dice';
        randomBtn.textContent = '🎲';
        randomBtn.title = 'Random palette';
        randomBtn.addEventListener('click', () => this._geRandomize());
        toolbar.appendChild(hint);
        toolbar.appendChild(randomBtn);
        ge.appendChild(toolbar);

        // Track: canvas + handle rail (unified visual block)
        const track = document.createElement('div');
        track.className = 'ge-track';

        const canvas = document.createElement('canvas');
        canvas.className = 'ge-canvas';
        canvas.height = 28;
        track.appendChild(canvas);
        this._geCanvas = canvas;

        const rail = document.createElement('div');
        rail.className = 'ge-rail';
        track.appendChild(rail);
        this._geRail = rail;

        // Click on track (not on a handle) → add a new color stop
        let trackDownPos = null;
        track.addEventListener('pointerdown', (e) => {
            if (e.target.closest('.ge-handle')) return;
            trackDownPos = { x: e.clientX, y: e.clientY };
        });
        track.addEventListener('pointerup', (e) => {
            if (!trackDownPos) return;
            const dx = Math.abs(e.clientX - trackDownPos.x);
            const dy = Math.abs(e.clientY - trackDownPos.y);
            trackDownPos = null;
            if (dx > 4 || dy > 4) return;
            if (e.target.closest('.ge-handle')) return;
            const rect = canvas.getBoundingClientRect();
            const p = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            this._geAddStop(p);
        });

        ge.appendChild(track);

        // Inline color editor (hidden by default, appears below rail)
        const editor = document.createElement('div');
        editor.className = 'ge-editor';
        ge.appendChild(editor);
        this._geEditor = editor;
        this._geActiveIdx = -1;

        wrap.appendChild(ge);
        this._geContainer = ge;

        // Build initial handles
        this._geRebuild();

        return wrap;
    }

    // ── Gradient editor: coordinate conversion ──

    _toHex(c) {
        return '#' +
            c.r.toString(16).padStart(2, '0') +
            c.g.toString(16).padStart(2, '0') +
            c.b.toString(16).padStart(2, '0');
    }

    _fromHex(hex) {
        const h = hex.replace('#', '');
        return {
            r: parseInt(h.slice(0, 2), 16) || 0,
            g: parseInt(h.slice(2, 4), 16) || 0,
            b: parseInt(h.slice(4, 6), 16) || 0,
            a: 255,
        };
    }

    // Numerically invert a monotone-increasing fn: find x s.t. fn(x) ≈ y.
    _invertGradFn(y) {
        const fn = this.settings.gradientFunction;
        let lo = 0, hi = 1;
        for (let k = 0; k < 40; k++) {
            const mid = (lo + hi) / 2;
            if (fn(mid) < y) lo = mid; else hi = mid;
        }
        return (lo + hi) / 2;
    }

    _weightToVisual(wf) {
        if (wf <= 0) return 0;
        if (wf >= 1) return 1;
        return this._invertGradFn(wf);
    }

    _visualToWeight(vf) {
        return this.settings.gradientFunction(Math.max(0, Math.min(1, vf)));
    }

    // Compute visual positions (0..1) for all color stops from weights.
    _geGetPositions() {
        const { weights } = this.settings;
        const total = weights.reduce((a, b) => a + b, 0);
        if (total === 0) return this.settings.colors.map((_, i, a) => i / Math.max(1, a.length - 1));
        const positions = [0];
        let cum = 0;
        for (let i = 0; i < weights.length; i++) {
            cum += weights[i];
            positions.push(this._weightToVisual(cum / total));
        }
        return positions;
    }

    // Write visual positions back as weights (preserving total).
    _geSetPositionsAsWeights(positions) {
        const oldTotal = this.settings.weights.reduce((a, b) => a + b, 0) || 1;
        const wFracs = positions.map(p => this._visualToWeight(p));
        const newWeights = [];
        for (let i = 0; i < wFracs.length - 1; i++) {
            newWeights.push(Math.max(1e-6, (wFracs[i + 1] - wFracs[i]) * oldTotal));
        }
        this.settings.weights = newWeights;
    }

    // ── Gradient editor: rendering ──

    _geRebuild() {
        this._geRail.classList.remove('ge-rail--dragging');
        const displayW = this._geCanvas.offsetWidth;
        if (displayW > 0) this._geCanvas.width = displayW;
        this._geRedraw();
        this._geRebuildHandles();
        // Keep editor open if the selected stop still exists
        if (this._geActiveIdx >= 0 && this._geActiveIdx < this.settings.colors.length) {
            this._geShowEditor(this._geActiveIdx);
        } else {
            this._geCloseEditor();
        }
    }

    _geRedraw() {
        drawGradientPreview(this._geCanvas, this.settings.colors, this.settings.weights, this.settings.gradientFunction);
    }

    _geRebuildHandles() {
        const rail = this._geRail;
        rail.innerHTML = '';

        const positions = this._geGetPositions();
        const colors = this.settings.colors;

        colors.forEach((color, i) => {
            const isDraggable = i > 0 && i < colors.length - 1;
            const pos = positions[i];

            const handle = document.createElement('div');
            handle.className = 'ge-handle';
            if (!isDraggable) handle.classList.add('ge-handle--endpoint');
            if (i === this._geActiveIdx) handle.classList.add('ge-handle--active');
            handle.style.left = (pos * 100) + '%';
            handle.style.setProperty('--stop-color', this._toHex(color));
            handle.setAttribute('role', 'slider');
            handle.setAttribute('aria-label', `Color stop ${i + 1}`);
            handle.setAttribute('tabindex', '0');

            // ── Pointer-based drag + click (unified mouse & touch) ──
            let dragState = null;

            handle.addEventListener('pointerdown', (e) => {
                if (e.button !== 0) return;
                e.preventDefault();
                e.stopPropagation();
                handle.setPointerCapture(e.pointerId);
                dragState = {
                    pointerId: e.pointerId,
                    startX: e.clientX,
                    dragging: false,
                    curIdx: i,
                    draggedColor: { ...this.settings.colors[i] },
                    positions: null, // snapshotted on first move
                    handles: null,   // data-index -> DOM-handle mapping during drag
                };
            });

            handle.addEventListener('pointermove', (e) => {
                if (!dragState || dragState.pointerId !== e.pointerId) return;
                if (!isDraggable) return;

                const dx = Math.abs(e.clientX - dragState.startX);
                if (!dragState.dragging && dx < 4) return;

                if (!dragState.dragging) {
                    dragState.dragging = true;
                    dragState.positions = this._geGetPositions();
                    dragState.handles = Array.from(rail.children);
                    rail.classList.add('ge-rail--dragging');
                    handle.classList.add('ge-handle--dragging');
                    this._geCloseEditor();
                }

                const railRect = rail.getBoundingClientRect();
                const rawP = Math.max(0, Math.min(1, (e.clientX - railRect.left) / railRect.width));
                const margin = 6 / (railRect.width || 300);
                const clampedP = Math.max(margin, Math.min(1 - margin, rawP));

                const colors = this.settings.colors;
                const pos = dragState.positions;
                const handles = dragState.handles;
                let ci = dragState.curIdx;

                // Bubble left: crossed neighbor shifts right to its previous slot.
                while (ci > 1 && clampedP < pos[ci - 1]) {
                    pos[ci] = pos[ci - 1];
                    [colors[ci], colors[ci - 1]] = [colors[ci - 1], colors[ci]];
                    [handles[ci], handles[ci - 1]] = [handles[ci - 1], handles[ci]];
                    ci--;
                }

                // Bubble right: crossed neighbor shifts left.
                while (ci < colors.length - 2 && clampedP > pos[ci + 1]) {
                    pos[ci] = pos[ci + 1];
                    [colors[ci], colors[ci + 1]] = [colors[ci + 1], colors[ci]];
                    [handles[ci], handles[ci + 1]] = [handles[ci + 1], handles[ci]];
                    ci++;
                }

                // Dragged stop always stays exactly under pointer.
                pos[ci] = clampedP;

                dragState.curIdx = ci;

                // Ensure dragged color stays consistent
                colors[ci] = { ...dragState.draggedColor };

                // Derive weights from the authoritative positions array
                this._geSetPositionsAsWeights(pos);
                this._geRedraw();
                this._geNotify();

                // Update ALL handle positions and colors
                for (let j = 0; j < handles.length; j++) {
                    handles[j].style.left = (pos[j] * 100) + '%';
                    handles[j].style.setProperty('--stop-color', this._toHex(colors[j]));
                }
            });

            handle.addEventListener('pointerup', (e) => {
                if (!dragState || dragState.pointerId !== e.pointerId) return;
                const wasDrag = dragState.dragging;
                rail.classList.remove('ge-rail--dragging');
                handle.classList.remove('ge-handle--dragging');
                dragState = null;

                if (!wasDrag) {
                    // Click → toggle editor for this stop
                    if (this._geActiveIdx === i) {
                        this._geCloseEditor();
                    } else {
                        this._geShowEditor(i);
                    }
                } else {
                    // After drag, rebuild to ensure full consistency
                    this._geRebuild();
                }
            });

            handle.addEventListener('lostpointercapture', () => {
                if (dragState) {
                    rail.classList.remove('ge-rail--dragging');
                    handle.classList.remove('ge-handle--dragging');
                    dragState = null;
                    this._geRebuild();
                }
            });

            // Keyboard support: nudge with arrow keys
            handle.addEventListener('keydown', (e) => {
                if (!isDraggable) return;
                const step = e.shiftKey ? 0.05 : 0.01;
                let delta = 0;
                if (e.key === 'ArrowLeft') delta = -step;
                else if (e.key === 'ArrowRight') delta = step;
                else if (e.key === 'Delete' || e.key === 'Backspace') {
                    if (colors.length > 2) { this._geRemoveStop(i); }
                    e.preventDefault();
                    return;
                }
                if (delta === 0) return;
                e.preventDefault();

                const curPositions = this._geGetPositions();
                const margin = 0.01;
                const leftBound = curPositions[i - 1] + margin;
                const rightBound = curPositions[i + 1] - margin;
                curPositions[i] = Math.max(leftBound, Math.min(rightBound, curPositions[i] + delta));
                this._geSetPositionsAsWeights(curPositions);
                this._geRebuild();
                this._geNotify();
            });

            rail.appendChild(handle);
        });
    }

    // ── Gradient editor: inline color editor ──

    _geShowEditor(idx) {
        this._geActiveIdx = idx;
        const editor = this._geEditor;
        editor.innerHTML = '';
        editor.classList.add('ge-editor--open');

        const colors = this.settings.colors;
        const color = colors[idx];

        // Highlight active handle
        this._geRail.querySelectorAll('.ge-handle').forEach((h, j) => {
            h.classList.toggle('ge-handle--active', j === idx);
        });

        // Header row
        const header = document.createElement('div');
        header.className = 'ge-editor__header';
        const title = document.createElement('span');
        title.textContent = idx === 0 ? 'First stop' : idx === colors.length - 1 ? 'Last stop' : `Stop ${idx + 1} of ${colors.length}`;
        header.appendChild(title);
        const closeBtn = document.createElement('button');
        closeBtn.className = 'ge-editor__close';
        closeBtn.textContent = '×';
        closeBtn.title = 'Close editor';
        closeBtn.addEventListener('click', () => this._geCloseEditor());
        header.appendChild(closeBtn);
        editor.appendChild(header);

        // Color picker + hex input row
        const colorRow = document.createElement('div');
        colorRow.className = 'ge-editor__color-row';

        const colorInp = document.createElement('input');
        colorInp.type = 'color';
        colorInp.className = 'ge-editor__picker';
        colorInp.value = this._toHex(color);

        const hexInp = document.createElement('input');
        hexInp.type = 'text';
        hexInp.className = 'ge-editor__hex';
        hexInp.value = this._toHex(color).toUpperCase();
        hexInp.maxLength = 7;
        hexInp.spellcheck = false;
        hexInp.placeholder = '#RRGGBB';

        const applyColor = (hex) => {
            const c = this._fromHex(hex);
            this.settings.colors[idx] = c;
            const handle = this._geRail.children[idx];
            if (handle) handle.style.setProperty('--stop-color', this._toHex(c));
            this._geRedraw();
            this._geNotify();
        };

        colorInp.addEventListener('input', () => {
            hexInp.value = colorInp.value.toUpperCase();
            applyColor(colorInp.value);
        });

        hexInp.addEventListener('input', () => {
            const v = hexInp.value.trim();
            if (/^#[0-9a-fA-F]{6}$/.test(v)) {
                colorInp.value = v;
                applyColor(v);
            }
        });

        hexInp.addEventListener('blur', () => {
            hexInp.value = this._toHex(this.settings.colors[idx]).toUpperCase();
        });

        colorRow.appendChild(colorInp);
        colorRow.appendChild(hexInp);

        // Delete button (if > 2 stops)
        if (colors.length > 2) {
            const delBtn = document.createElement('button');
            delBtn.className = 'dash-btn dash-btn--danger ge-editor__delete';
            delBtn.textContent = '✕ Remove';
            delBtn.addEventListener('click', () => this._geRemoveStop(idx));
            colorRow.appendChild(delBtn);
        }

        editor.appendChild(colorRow);
    }

    _geCloseEditor() {
        this._geActiveIdx = -1;
        this._geEditor.classList.remove('ge-editor--open');
        this._geEditor.innerHTML = '';
        if (this._geRail) {
            this._geRail.querySelectorAll('.ge-handle').forEach(h => h.classList.remove('ge-handle--active'));
        }
    }

    // ── Gradient editor: actions ──

    _geAddStop(visualPos) {
        const colors = this.settings.colors;
        const weights = this.settings.weights;
        const total = weights.reduce((a, b) => a + b, 0);

        const wPos = this._visualToWeight(visualPos) * total;

        let cumW = 0, k = 0;
        for (; k < weights.length - 1; k++) {
            if (cumW + weights[k] > wPos) break;
            cumW += weights[k];
        }

        const t = weights[k] > 0 ? (wPos - cumW) / weights[k] : 0.5;
        const c0 = colors[k], c1 = colors[k + 1];
        const newColor = {
            r: Math.round(c0.r + (c1.r - c0.r) * t),
            g: Math.round(c0.g + (c1.g - c0.g) * t),
            b: Math.round(c0.b + (c1.b - c0.b) * t),
            a: 255,
        };

        const leftW = wPos - cumW;
        const rightW = weights[k] - leftW;
        colors.splice(k + 1, 0, newColor);
        weights.splice(k, 1, leftW, rightW);

        this._geRebuild();
        this._geNotify();
        this._geShowEditor(k + 1);
    }

    _geRemoveStop(idx) {
        const weights = this.settings.weights;
        if (idx === 0) {
            weights.splice(0, 1);
        } else if (idx >= weights.length) {
            weights.splice(idx - 1, 1);
        } else {
            weights[idx - 1] += weights[idx];
            weights.splice(idx, 1);
        }
        this.settings.colors.splice(idx, 1);
        this._geCloseEditor();
        this._geRebuild();
        this._geNotify();
    }

    _geRandomize() {
        const palette = randomPalette();
        this.settings.colors = palette.colors;
        this.settings.weights = palette.weights;
        this.settings.colorPeriod = palette.colorPeriod;
        if (this._fields['colorPeriod_slider']) this._fields['colorPeriod_slider'].value = palette.colorPeriod;
        if (this._fields['colorPeriod_num']) this._fields['colorPeriod_num'].value = palette.colorPeriod;
        this._geCloseEditor();
        this._geRebuild();
        this._geNotify();
    }

    _geNotify() {
        this.settings._colorVersion = (this.settings._colorVersion || 0) + 1;
        this.state.dirty = true;
    }

    // Called by reset handler in action bar.
    _onStopsChanged() {
        this._geCloseEditor();
        this._geRebuild();
        this._geNotify();
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

        const shareBtn = document.createElement('button');
        shareBtn.className = 'dash-btn action-btn';
        shareBtn.textContent = '🔗 Share link';
        shareBtn.addEventListener('click', () => {
            const hash = encodeShareHash(this.settings, this.state);
            const url = `${location.origin}${location.pathname}#${hash}`;
            navigator.clipboard.writeText(url).then(() => {
                shareBtn.textContent = 'Copied ✓';
                setTimeout(() => { shareBtn.textContent = '🔗 Share link'; }, 2000);
            }).catch(() => {
                // Fallback: update the URL bar and let the user copy manually
                location.hash = hash;
                shareBtn.textContent = 'Link in URL bar ✓';
                setTimeout(() => { shareBtn.textContent = '🔗 Share link'; }, 2500);
            });
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
        bar.appendChild(shareBtn);
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

    _updateHelp() {
        if (!this._helpEl) return;
        this._helpEl.innerHTML = 'Drag=Pan &nbsp;·&nbsp; Scroll/W/S=Zoom &nbsp;·&nbsp; Pinch=Zoom &nbsp;·&nbsp; Arrows=Pan &nbsp;·&nbsp; ☰=Toggle panel &nbsp;·&nbsp; F11=Fullscreen';
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
