/**
 * controls.js — Input handling: mouse, keyboard, and touch.
 *
 * Mutates a shared state object and sets state.dirty = true on changes.
 */

export class Controls {
    constructor(canvas, state, settings) {
        this.canvas = canvas;
        this.state = state;
        this.settings = settings;

        // Keyboard state
        this._keys = {};

        // Mouse drag state
        this._dragging = false;
        this._dragStartX = 0;
        this._dragStartY = 0;
        this._dragStartCenterX = 0;
        this._dragStartCenterY = 0;
        this._dragMoved = false;

        // Touch state
        this._activeTouches = new Map();
        this._pinchStartDist = 0;
        this._pinchStartMid = { x: 0, y: 0 };
        this._pinchStartCenter = { x: 0, y: 0 };
        this._pinchStartViewport = 0;
        this._singleTouchStart = null;
        this._singleTouchCenter = null;

        this._bindEvents();
    }

    _bindEvents() {
        const c = this.canvas;

        // --- Keyboard ---
        this._onKeyDown = (e) => this._handleKeyDown(e);
        this._onKeyUp = (e) => this._handleKeyUp(e);
        window.addEventListener('keydown', this._onKeyDown);
        window.addEventListener('keyup', this._onKeyUp);

        // --- Mouse ---
        this._onWheel = (e) => this._handleWheel(e);
        this._onMouseDown = (e) => this._handleMouseDown(e);
        this._onMouseMove = (e) => this._handleMouseMove(e);
        this._onMouseUp = (e) => this._handleMouseUp(e);
        c.addEventListener('wheel', this._onWheel, { passive: false });
        c.addEventListener('mousedown', this._onMouseDown);
        window.addEventListener('mousemove', this._onMouseMove);
        window.addEventListener('mouseup', this._onMouseUp);

        // --- Touch ---
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

        switch (e.key) {
            case 'm':
            case 'M':
                s.baseMaxIter = Math.ceil(s.baseMaxIter * this.settings.maxIterAdjustFactor);
                s.dirty = true;
                break;
            case 'n':
            case 'N':
                s.baseMaxIter = Math.max(10, Math.ceil(s.baseMaxIter / this.settings.maxIterAdjustFactor));
                s.dirty = true;
                break;
            case 'i':
            case 'I':
                if (this._onToggleOverlay) this._onToggleOverlay();
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

    _handleKeyUp(e) {
        this._keys[e.key] = false;
    }

    _toggleFullscreen() {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen().catch(() => {});
        }
    }

    /**
     * Called once per frame to apply held-key actions.
     */
    update() {
        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        let changed = false;

        // Pan with arrow keys
        if (this._keys['ArrowLeft'])  { s.centerX -= vpX * this.settings.panSpeed; changed = true; }
        if (this._keys['ArrowRight']) { s.centerX += vpX * this.settings.panSpeed; changed = true; }
        if (this._keys['ArrowUp'])    { s.centerY -= s.viewportSizeY * this.settings.panSpeed; changed = true; }
        if (this._keys['ArrowDown'])  { s.centerY += s.viewportSizeY * this.settings.panSpeed; changed = true; }

        // Zoom with W / S
        if (this._keys['w'] || this._keys['W']) { s.viewportSizeY *= this.settings.keyZoomSpeed; changed = true; }
        if (this._keys['s'] || this._keys['S']) { s.viewportSizeY /= this.settings.keyZoomSpeed; changed = true; }

        if (changed) s.dirty = true;
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
        this._dragStartCenterX = this.state.centerX;
        this._dragStartCenterY = this.state.centerY;
    }

    _handleMouseMove(e) {
        if (!this._dragging) return;
        const dx = e.clientX - this._dragStartX;
        const dy = e.clientY - this._dragStartY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) this._dragMoved = true;

        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        s.centerX = this._dragStartCenterX - (dx / this.canvas.clientWidth) * vpX;
        s.centerY = this._dragStartCenterY - (dy / this.canvas.clientHeight) * s.viewportSizeY;
        s.dirty = true;
    }

    _handleMouseUp(e) {
        if (!this._dragging) return;
        this._dragging = false;

        // Click-to-center if there was no significant drag
        if (!this._dragMoved) {
            const s = this.state;
            const aspect = this.canvas.width / this.canvas.height;
            const vpX = s.viewportSizeY * aspect;

            const normX = e.clientX / this.canvas.clientWidth;
            const normY = e.clientY / this.canvas.clientHeight;

            s.centerX = s.centerX + (normX - 0.5) * vpX;
            s.centerY = s.centerY + (normY - 0.5) * s.viewportSizeY;
            s.dirty = true;
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
            this._singleTouchCenter = { x: this.state.centerX, y: this.state.centerY };
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

        if (this._activeTouches.size === 2) {
            this._handlePinch();
        } else if (this._activeTouches.size === 1) {
            this._handleSingleTouchDrag();
        }
    }

    _handleTouchEnd(e) {
        for (const t of e.changedTouches) {
            this._activeTouches.delete(t.identifier);
        }
        if (this._activeTouches.size === 1) {
            const t = [...this._activeTouches.values()][0];
            this._singleTouchStart = { x: t.x, y: t.y };
            this._singleTouchCenter = { x: this.state.centerX, y: this.state.centerY };
        }
    }

    _initPinch() {
        const pts = [...this._activeTouches.values()];
        this._pinchStartDist = this._dist(pts[0], pts[1]);
        this._pinchStartMid = this._mid(pts[0], pts[1]);
        this._pinchStartCenter = { x: this.state.centerX, y: this.state.centerY };
        this._pinchStartViewport = this.state.viewportSizeY;
    }

    _handlePinch() {
        const pts = [...this._activeTouches.values()];
        const dist = this._dist(pts[0], pts[1]);
        const mid = this._mid(pts[0], pts[1]);

        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;

        // Zoom: inverse ratio of distance change
        const factor = this._pinchStartDist / dist;
        s.viewportSizeY = this._pinchStartViewport * factor;
        const vpX = s.viewportSizeY * aspect;

        // Pan: delta of midpoint
        const dx = (mid.x - this._pinchStartMid.x) / this.canvas.clientWidth;
        const dy = (mid.y - this._pinchStartMid.y) / this.canvas.clientHeight;
        const startVpX = this._pinchStartViewport * aspect;
        s.centerX = this._pinchStartCenter.x - dx * startVpX;
        s.centerY = this._pinchStartCenter.y - dy * this._pinchStartViewport;

        s.dirty = true;
    }

    _handleSingleTouchDrag() {
        if (!this._singleTouchStart) return;
        const t = [...this._activeTouches.values()][0];
        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        const dx = t.x - this._singleTouchStart.x;
        const dy = t.y - this._singleTouchStart.y;

        s.centerX = this._singleTouchCenter.x - (dx / this.canvas.clientWidth) * vpX;
        s.centerY = this._singleTouchCenter.y - (dy / this.canvas.clientHeight) * s.viewportSizeY;
        s.dirty = true;
    }

    // --- Zoom at screen point ---

    _zoomAtPoint(clientX, clientY, factor) {
        const s = this.state;
        const aspect = this.canvas.width / this.canvas.height;
        const vpX = s.viewportSizeY * aspect;

        const normX = clientX / this.canvas.clientWidth;
        const normY = clientY / this.canvas.clientHeight;

        const mouseRe = s.centerX + (normX - 0.5) * vpX;
        const mouseIm = s.centerY + (normY - 0.5) * s.viewportSizeY;

        s.viewportSizeY *= factor;
        const newVpX = s.viewportSizeY * aspect;

        s.centerX = mouseRe - (normX - 0.5) * newVpX;
        s.centerY = mouseIm - (normY - 0.5) * s.viewportSizeY;
        s.dirty = true;
    }

    // --- Utilities ---

    _dist(a, b) {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    _mid(a, b) {
        return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    }

    // --- Callbacks (set from main.js) ---
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
