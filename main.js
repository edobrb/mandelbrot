/**
 * main.js — Entry point, initialization, render loop.
 * Uses perturbation theory with arbitrary-precision reference orbits.
 */

import { defaultSettings } from './settings.js';
import { buildColorLUT } from './colors.js';
import { Renderer } from './renderer.js';
import { Controls } from './controls.js';
import { UI } from './ui.js';
import { BigFloat } from './bigfloat.js';
import { computeReferenceOrbit } from './perturbation.js';
import { loadSavedSettings } from './storage.js';

// ---------- Dynamic max iterations ----------

function computeDynamicMaxIter(viewportSizeY, baseMaxIter) {
    const val = Math.sqrt(2 * Math.sqrt(Math.abs(1 - Math.sqrt(5 / viewportSizeY)))) * (baseMaxIter / 10);
    return Math.max(10, Math.ceil(val));
}

// ---------- Screenshot ----------

async function takeScreenshot(renderer) {
    const blob = await renderer.captureScreenshot();
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mandelbrot_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ---------- Boot ----------

async function main() {
    const errorEl = document.getElementById('error-message');
    const canvas = document.getElementById('mandelbrot-canvas');

    if (!navigator.gpu) {
        errorEl.textContent = 'WebGPU is not supported in your browser. Please use a recent version of Chrome, Edge, or Firefox Nightly.';
        errorEl.classList.remove('hidden');
        canvas.classList.add('hidden');
        return;
    }

    const settings = { ...defaultSettings };

    // Apply saved settings over defaults
    const saved = loadSavedSettings();
    if (saved) {
        if (saved.colors)              settings.colors = saved.colors;
        if (saved.weights)             settings.weights = saved.weights;
        if (saved.colorPeriod)         settings.colorPeriod = saved.colorPeriod;
        if (saved.zoomSpeed)           settings.zoomSpeed = saved.zoomSpeed;
        if (saved.panSpeed)            settings.panSpeed = saved.panSpeed;
        if (saved.keyZoomSpeed)        settings.keyZoomSpeed = saved.keyZoomSpeed;
        if (saved.maxIterAdjustFactor) settings.maxIterAdjustFactor = saved.maxIterAdjustFactor;
    }

    // Shared state — center is tracked in both BigFloat (precision) and f64 (display)
    const state = {
        centerX: settings.initialCenterX,
        centerY: settings.initialCenterY,
        centerBF_X: BigFloat.fromNumber(settings.initialCenterX),
        centerBF_Y: BigFloat.fromNumber(settings.initialCenterY),
        viewportSizeY: settings.initialViewportSizeY,
        baseMaxIter: saved?.baseMaxIter ?? settings.initialMaxIter,
        maxIter: saved?.baseMaxIter ?? settings.initialMaxIter,
        maxiterMode: saved?.maxiterMode ?? settings.maxiterMode,
        smartphoneMode: saved?.smartphoneMode ?? false,
        dirty: true,
        refOrbitDirty: true,
        fps: 0,
        width: 0,
        height: 0,
    };

    // Initialize renderer
    const renderer = new Renderer(canvas);
    try {
        await renderer.init();
    } catch (err) {
        errorEl.textContent = `WebGPU initialization failed: ${err.message}`;
        errorEl.classList.remove('hidden');
        canvas.classList.add('hidden');
        return;
    }

    const controls = new Controls(canvas, state, settings);

    function navigateTo(x, y, viewport) {
        state.centerBF_X = BigFloat.fromString(x);
        state.centerBF_Y = BigFloat.fromString(y);
        state.centerX = state.centerBF_X.toNumber();
        state.centerY = state.centerBF_Y.toNumber();
        state.viewportSizeY = viewport;
        state.refOrbitDirty = true;
        state.dirty = true;
    }

    const ui = new UI(
        document.getElementById('ui-container'),
        settings,
        state,
        {
            navigateTo,
            takeScreenshot: () => takeScreenshot(renderer),
            defaultSettings,
        },
    );
    controls.onToggleOverlay(() => ui.toggle());
    controls.onScreenshot(() => takeScreenshot(renderer));

    // ---------- Reference orbit cache ----------

    let refOrbit = null;
    let refOrbitMaxIter = -1;
    let lastLUTMaxIter = -1;
    let lastColorPeriod = -1;
    let lastColorVersion = -1;

    function updateRefOrbit(maxIter) {
        const t0 = performance.now();
        refOrbit = computeReferenceOrbit(state.centerBF_X, state.centerBF_Y, maxIter);
        refOrbitMaxIter = maxIter;
        renderer.updateRefOrbit(refOrbit.re, refOrbit.im, refOrbit.length);
        state.refOrbitDirty = false;
        const dt = performance.now() - t0;
        if (dt > 50) console.log(`Reference orbit: ${refOrbit.length} iters in ${dt.toFixed(1)}ms`);
    }

    // ---------- Sizing ----------

    function handleResize() {
        const dpr = window.devicePixelRatio || 1;
        const w = Math.floor(canvas.clientWidth * dpr);
        const h = Math.floor(canvas.clientHeight * dpr);
        if (w !== state.width || h !== state.height) {
            state.width = w;
            state.height = h;
            renderer.resize(w, h);
            state.dirty = true;
        }
    }

    window.addEventListener('resize', () => { state.dirty = true; });

    // ---------- Render loop ----------

    let fpsFrames = 0;
    let fpsTime = performance.now();

    function loop() {
        requestAnimationFrame(loop);

        handleResize();
        controls.update();

        // Compute effective maxIter
        if (state.maxiterMode === 'Dynamic') {
            state.maxIter = computeDynamicMaxIter(state.viewportSizeY, state.baseMaxIter);
        } else {
            state.maxIter = state.baseMaxIter;
        }

        // Rebuild color LUT when colorPeriod or gradient colors change
        const colorVersion = settings._colorVersion || 0;
        if (state.maxIter !== lastLUTMaxIter || settings.colorPeriod !== lastColorPeriod || colorVersion !== lastColorVersion) {
            const lut = buildColorLUT(settings.colorPeriod, settings.colors, settings.weights, settings.gradientFunction);
            renderer.updateColorLUT(lut, settings.colorPeriod);
            lastLUTMaxIter = state.maxIter;
            lastColorPeriod = settings.colorPeriod;
            lastColorVersion = colorVersion;
            state.refOrbitDirty = true;
            state.dirty = true;
        }

        // Recompute reference orbit when needed
        if (state.refOrbitDirty || refOrbitMaxIter !== state.maxIter) {
            updateRefOrbit(state.maxIter);
            state.dirty = true;
        }

        // Only render when something changed
        if (state.dirty) {
            const aspect = state.width / state.height;
            const vpX = state.viewportSizeY * aspect;

            renderer.render(vpX, state.viewportSizeY, refOrbit.length, state.maxIter, settings.colorPeriod);
            state.dirty = false;
        }

        // FPS
        fpsFrames++;
        const now = performance.now();
        if (now - fpsTime >= 1000) {
            state.fps = fpsFrames;
            fpsFrames = 0;
            fpsTime = now;
        }

        ui.update(state);
    }

    requestAnimationFrame(loop);
}

main();
