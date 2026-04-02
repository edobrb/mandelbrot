/**
 * main.js — Entry point, initialization, render loop.
 */

import { defaultSettings } from './settings.js';
import { buildColorLUT } from './colors.js';
import { Renderer } from './renderer.js';
import { Controls } from './controls.js';
import { UI } from './ui.js';

// ---------- Dynamic max iterations ----------

function computeDynamicMaxIter(viewportSizeY, baseMaxIter) {
    const val = Math.sqrt(2 * Math.sqrt(Math.abs(1 - Math.sqrt(5 / viewportSizeY)))) * (baseMaxIter / 10);
    return Math.max(10, Math.ceil(val));
}

// ---------- Screenshot ----------

function takeScreenshot(canvas) {
    const temp = document.createElement('canvas');
    temp.width = canvas.width;
    temp.height = canvas.height;
    const ctx = temp.getContext('2d');
    ctx.drawImage(canvas, 0, 0);
    temp.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mandelbrot_${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 'image/png');
}

// ---------- Boot ----------

async function main() {
    const errorEl = document.getElementById('error-message');
    const canvas = document.getElementById('mandelbrot-canvas');

    // Check WebGPU support
    if (!navigator.gpu) {
        errorEl.textContent = 'WebGPU is not supported in your browser. Please use a recent version of Chrome, Edge, or Firefox Nightly.';
        errorEl.classList.remove('hidden');
        canvas.classList.add('hidden');
        return;
    }

    // Shared state
    const settings = { ...defaultSettings };
    const state = {
        centerX: settings.initialCenterX,
        centerY: settings.initialCenterY,
        viewportSizeY: settings.initialViewportSizeY,
        baseMaxIter: settings.initialMaxIter,
        maxIter: settings.initialMaxIter,
        maxiterMode: settings.maxiterMode,
        dirty: true,
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

    // Initialize controls
    const controls = new Controls(canvas, state, settings);

    // Initialize UI
    const ui = new UI(document.getElementById('ui-container'), settings);
    controls.onToggleOverlay(() => ui.toggle());
    controls.onScreenshot(() => takeScreenshot(canvas));

    // ---------- Sizing ----------

    let lastLUTMaxIter = -1;

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

        // Rebuild color LUT when maxIter changes
        if (state.maxIter !== lastLUTMaxIter) {
            const lut = buildColorLUT(state.maxIter, settings.colors, settings.weights, settings.gradientFunction);
            renderer.updateColorLUT(lut, state.maxIter);
            lastLUTMaxIter = state.maxIter;
            state.dirty = true;
        }

        // Only render when something changed
        if (state.dirty) {
            const aspect = state.width / state.height;
            const vpX = state.viewportSizeY * aspect;
            const x0 = state.centerX - vpX / 2;
            const x1 = state.centerX + vpX / 2;
            const y0 = state.centerY - state.viewportSizeY / 2;
            const y1 = state.centerY + state.viewportSizeY / 2;

            renderer.render(x0, x1, y0, y1, state.maxIter);
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
