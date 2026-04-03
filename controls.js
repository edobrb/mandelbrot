/**
 * controls.js — Input handling: mouse, keyboard, and touch.
 *
 * Mutates a shared state object and sets state.dirty = true on changes.
 * Center coordinates are tracked in BigFloat for arbitrary precision zoom.
 */

import { BigFloat } from './bigfloat.js';

export class Controls {
    constructor(canvas, state, settings) {
        this.canvas = canvas;
        this.state = state;
        this.settings = settings;

        this._keys = {};

        // Mouse drag state
        this._dragging = false;
        this._dragStartX = 0;
        this._dragStartY = 0;
        this._dragStartCenterBF_X = null;
        this._dragStartCenterBF_Y = null;
        this._dragMoved = false;

        // Touch state
        this._activeTouches = new Map();
        this._pinchStartDist = 0;
        this._pinchStartMid = { x: 0, y: 0 };
        this._pinchStartCenterBF_X = null;
        this._pinchStartCenterBF_Y = null;
        this._pinchStartViewport = 0;
        this._singleTouchStart = null;
        this._singleTouchCenterBF_X = null;
        this._singleTouchCenterBF_Y = null;

        this._bindEvents();
    }

    /**
     * Update the BigFloat center and sync f64 approximation.
     */
    _setCenter(bfX, bfY) {
        const s = this.state;
        s.centerBF_X = bfX;
        s.centerBF_Y = bfY;
        s.centerX = bfX.toNumber();
        s.centerY = bfY.toNumber();
        s.refOrbitDirty = true;
        s.dirty = true;
    }

    /**
     * Offset the BigFloat center by f64 delta values.
     */
    _offsetCenter(dx, dy) {
        const s = this.state;
        this._setCenter(
            s.centerBF_X.add(BigFloat.fromNumber(dx)),
            s.centerBF_Y.add(BigFloat.fromNumber(dy)),
        );
    }

    _bindEvents() {
        const c = this.canvas;

        this._onKeyDown = (e) => this._handleKeyDown(e);
        this._onKeyUp = (e) => this._handleKeyUp(e);
        window.addEventListener('keydown', this._onKeyDown);
        window.addEventListener('keyup', this._onKeyUp);

        this._onWheel = (e) => this._handleWheel(e);
        this._onMouseDown = (e) => this._handleMouseDown(e);
        this._onMouseMove = (e) => this._handleMouseMove(e);
        this._onMouseUp = (e) => this._handleMouseUp(e);
        c.addEventListener('wheel', this._onWheel, { passive: false });
        c.addEventListener('mousedown', this._onMouseDown);
        window.addEventListener('mousemove', this._onMouseMove);
        window.addEventListener('mouseup', this._onMouseUp);

        this._onTouchStart = (e) => this._handleTouchStart(e);
        this._onTouchMove = (e) => this._handleTouchMove(e);
        this._onTouchEnd = (e) => this._handleTouchEnd(e);
        c.addEventListener('touchstart', this._onTouchStart, { passive: false });
        c.addEventListener('touchmove', this._onTouchMove, { passive: false });
        c.addEventListener('touchend', this._onTouchEnd);
        c.addEventListener('touchcancel', this._onTouchEnd);
    }

    // --- Keyboard handlers ---

    _handleKeyDown(e) {
        this._keys[e.key] = true;
        const s = this.state;

        const tag = document.activeElement?.tagName;
        const isTyping = tag === 'INPUT' || tag === 'TEXTAREA';

        switch (e.key) {
            case 'i': case 'I':
                if (!isTyping && this._onToggleOverlay) this._onToggleOverlay();
                break;
            case 'F11':
                e.preventDefault();
                this._toggleFullscreen();
                break;
            case 'F12':
                e.preventDefault();
                if (this._onScreenshot) this._onScreenshot();
                break;
        }
    }

    _handleKeyUp(e) { this._keys[e.key] = false; }

    _toggleFullscreen() {
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen().catch(() => {});
    }

    update() {
        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        const tag = document.activeElement?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;

        if (this._keys['ArrowLeft'])  { this._offsetCenter(-vpX * this.settings.panSpeed, 0); }
        if (this._keys['ArrowRight']) { this._offsetCenter(vpX * this.settings.panSpeed, 0); }
        if (this._keys['ArrowUp'])    { this._offsetCenter(0, -s.viewportSizeY * this.settings.panSpeed); }
        if (this._keys['ArrowDown'])  { this._offsetCenter(0, s.viewportSizeY * this.settings.panSpeed); }

        if (this._keys['w'] || this._keys['W']) { s.viewportSizeY *= this.settings.keyZoomSpeed; s.dirty = true; }
        if (this._keys['s'] || this._keys['S']) { s.viewportSizeY /= this.settings.keyZoomSpeed; s.dirty = true; }
    }

    // --- Mouse handlers ---

    _handleWheel(e) {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1 / this.settings.zoomSpeed : this.settings.zoomSpeed;
        this._zoomAtPoint(e.clientX, e.clientY, factor);
    }

    _handleMouseDown(e) {
        if (e.button !== 0) return;
        this._dragging = true;
        this._dragMoved = false;
        this._dragStartX = e.clientX;
        this._dragStartY = e.clientY;
        this._dragStartCenterBF_X = this.state.centerBF_X.clone();
        this._dragStartCenterBF_Y = this.state.centerBF_Y.clone();
    }

    _handleMouseMove(e) {
        if (!this._dragging) return;
        const dx = e.clientX - this._dragStartX;
        const dy = e.clientY - this._dragStartY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) this._dragMoved = true;

        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        const complexDx = -(dx / this.canvas.clientWidth) * vpX;
        const complexDy = -(dy / this.canvas.clientHeight) * s.viewportSizeY;

        this._setCenter(
            this._dragStartCenterBF_X.add(BigFloat.fromNumber(complexDx)),
            this._dragStartCenterBF_Y.add(BigFloat.fromNumber(complexDy)),
        );
    }

    _handleMouseUp(e) {
        if (!this._dragging) return;
        this._dragging = false;

        if (!this._dragMoved) {
            this._zoomAtPoint(e.clientX, e.clientY, 1.0);
        }
    }

    // --- Touch handlers ---

    _handleTouchStart(e) {
        e.preventDefault();
        for (const t of e.changedTouches) {
            this._activeTouches.set(t.identifier, { x: t.clientX, y: t.clientY });
        }

        if (this._activeTouches.size === 1) {
            const t = [...this._activeTouches.values()][0];
            this._singleTouchStart = { x: t.x, y: t.y };
            this._singleTouchCenterBF_X = this.state.centerBF_X.clone();
            this._singleTouchCenterBF_Y = this.state.centerBF_Y.clone();
        } else if (this._activeTouches.size === 2) {
            this._initPinch();
        }
    }

    _handleTouchMove(e) {
        e.preventDefault();
        for (const t of e.changedTouches) {
            if (this._activeTouches.has(t.identifier)) {
                this._activeTouches.set(t.identifier, { x: t.clientX, y: t.clientY });
            }
        }

        if (this._activeTouches.size === 2) this._handlePinch();
        else if (this._activeTouches.size === 1) this._handleSingleTouchDrag();
    }

    _handleTouchEnd(e) {
        for (const t of e.changedTouches) {
            this._activeTouches.delete(t.identifier);
        }
        if (this._activeTouches.size === 1) {
            const t = [...this._activeTouches.values()][0];
            this._singleTouchStart = { x: t.x, y: t.y };
            this._singleTouchCenterBF_X = this.state.centerBF_X.clone();
            this._singleTouchCenterBF_Y = this.state.centerBF_Y.clone();
        }
    }

    _initPinch() {
        const pts = [...this._activeTouches.values()];
        this._pinchStartDist = this._dist(pts[0], pts[1]);
        this._pinchStartMid = this._mid(pts[0], pts[1]);
        this._pinchStartCenterBF_X = this.state.centerBF_X.clone();
        this._pinchStartCenterBF_Y = this.state.centerBF_Y.clone();
        this._pinchStartViewport = this.state.viewportSizeY;
    }

    _handlePinch() {
        const pts = [...this._activeTouches.values()];
        const dist = this._dist(pts[0], pts[1]);
        const mid = this._mid(pts[0], pts[1]);

        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;

        const factor = this._pinchStartDist / dist;
        s.viewportSizeY = this._pinchStartViewport * factor;

        const dx = (mid.x - this._pinchStartMid.x) / this.canvas.clientWidth;
        const dy = (mid.y - this._pinchStartMid.y) / this.canvas.clientHeight;
        const startVpX = this._pinchStartViewport * aspect;

        this._setCenter(
            this._pinchStartCenterBF_X.sub(BigFloat.fromNumber(dx * startVpX)),
            this._pinchStartCenterBF_Y.sub(BigFloat.fromNumber(dy * this._pinchStartViewport)),
        );
    }

    _handleSingleTouchDrag() {
        if (!this._singleTouchStart) return;
        const t = [...this._activeTouches.values()][0];
        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        const dx = t.x - this._singleTouchStart.x;
        const dy = t.y - this._singleTouchStart.y;

        this._setCenter(
            this._singleTouchCenterBF_X.sub(BigFloat.fromNumber((dx / this.canvas.clientWidth) * vpX)),
            this._singleTouchCenterBF_Y.sub(BigFloat.fromNumber((dy / this.canvas.clientHeight) * s.viewportSizeY)),
        );
    }

    // --- Zoom at screen point (BigFloat-aware) ---

    _zoomAtPoint(clientX, clientY, factor) {
        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        const normX = clientX / this.canvas.clientWidth;
        const normY = clientY / this.canvas.clientHeight;

        // Mouse position in complex plane = center + offset
        const offsetRe = (normX - 0.5) * vpX;
        const offsetIm = (normY - 0.5) * s.viewportSizeY;
        const mouseBF_Re = s.centerBF_X.add(BigFloat.fromNumber(offsetRe));
        const mouseBF_Im = s.centerBF_Y.add(BigFloat.fromNumber(offsetIm));

        s.viewportSizeY *= factor;
        const newVpX = s.viewportSizeY * aspect;

        // New center = mouse - new offset
        this._setCenter(
            mouseBF_Re.sub(BigFloat.fromNumber((normX - 0.5) * newVpX)),
            mouseBF_Im.sub(BigFloat.fromNumber((normY - 0.5) * s.viewportSizeY)),
        );
    }

    // --- Utilities ---
    _dist(a, b) { const dx = b.x - a.x, dy = b.y - a.y; return Math.sqrt(dx * dx + dy * dy); }
    _mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }

    onToggleOverlay(fn) { this._onToggleOverlay = fn; }
    onScreenshot(fn) { this._onScreenshot = fn; }

    destroy() {
        window.removeEventListener('keydown', this._onKeyDown);
        window.removeEventListener('keyup', this._onKeyUp);
        window.removeEventListener('mousemove', this._onMouseMove);
        window.removeEventListener('mouseup', this._onMouseUp);
        this.canvas.removeEventListener('wheel', this._onWheel);
        this.canvas.removeEventListener('mousedown', this._onMouseDown);
        this.canvas.removeEventListener('touchstart', this._onTouchStart);
        this.canvas.removeEventListener('touchmove', this._onTouchMove);
        this.canvas.removeEventListener('touchend', this._onTouchEnd);
        this.canvas.removeEventListener('touchcancel', this._onTouchEnd);
    }
}
