# Mandelbrot Visualizer — Implementation Documentation

## Architecture

The project is a single-page web app. Modules:

| File | Responsibility |
|---|---|
| `main.js` | Boot, render loop, resize handling |
| `renderer.js` | WebGPU device, pipeline, uniforms |
| `controls.js` | Mouse, keyboard, and touch input |
| `ui.js` | Settings dashboard DOM |
| `colors.js` | Gradient LUT builder and preview |
| `settings.js` | Default settings object |
| `storage.js` | localStorage persistence |
| `bigfloat.js` | Arbitrary-precision float arithmetic |
| `perturbation.js` | Reference orbit computation |
| `mandelbrot.wgsl` | Standard escape-time GPU shader |
| `mandelbrot_perturb.wgsl` | Perturbation theory GPU shader |

## Rendering

### Escape-time algorithm

Each fragment shader invocation computes one pixel:

```
z = 0
c = pixel coordinate in complex plane
while iterations < max_iter AND |z|² < 4:
    z = z² + c
    iterations++
```

The iteration count is used as an index into a precomputed color LUT uploaded as a GPU texture.

### Perturbation Theory

At high zoom levels, standard f64 arithmetic loses precision. The viewer uses perturbation theory: a single high-precision reference orbit is computed on the CPU using `BigFloat`, then each pixel computes only the small delta from that reference orbit using f32/f64 arithmetic on the GPU.

The reference orbit is recomputed whenever the center changes or `maxIter` increases.

### Color LUT

A `uint8[colorPeriod × 4]` RGBA array is built on the CPU from the configured color stops, weights, and gradient function, then uploaded to a WebGPU texture. The GPU kernel indexes into this texture using `iteration % colorPeriod`.

### Dynamic Max Iterations

When `maxiterMode` is `Dynamic`:

```
maxiter = sqrt(2 * sqrt(|1 - sqrt(5 / viewportSizeY)|)) * (baseMaxIter / 10)
```

This scales detail automatically as the user zooms deeper.

## Input Handling (`controls.js`)

### Mouse

- **Mousedown** captures drag start position and a `BigFloat` clone of the current center.
- **Mousemove** computes delta in pixels, converts to complex-plane offset, and applies it to the captured center clone.
- **Mouseup** with no drag movement triggers `_zoomAtPoint`.
- **Wheel** calls `_zoomAtPoint` with a zoom factor derived from `deltaY`.

### Keyboard

Arrow keys and W/S are checked each frame in `Controls.update()` (called from the render loop), so holding a key produces smooth continuous movement.

### Touch

- **Single touch** — same panning logic as mouse drag.
- **Two-finger pinch** — recorded at `touchstart`; each `touchmove` computes the ratio of current to initial finger distance and applies it as a zoom factor, while the midpoint translation is applied as a pan.

## UI (`ui.js`)

The settings dashboard is a fixed-position panel built entirely in JavaScript. Sections:

- **Live stats** — updated every frame via `UI.update()`.
- **Rendering** — maxiter mode segmented control + max iterations slider.
- **Colors** — color period slider, gradient canvas preview, color stop pickers, weights.
- **Navigation** — zoom/pan speed sliders, **smartphone mode** toggle.
- **Bookmarks** — named location save/restore.
- **Action bar** — screenshot, save settings, reset defaults.

### Smartphone Mode

Toggled via the **Navigation → Smartphone mode** segmented control (Off / On). When enabled:

- A floating **☰** button appears (bottom-right corner) that toggles the panel. This is the only way to show/hide the panel without a keyboard.
- The help text at the bottom of the panel switches from keyboard shortcuts to touch instructions (`Drag=Pan · Pinch=Zoom · ☰=Toggle panel`).
- The state is persisted to `localStorage` via **Save settings**.

## Persistence (`storage.js`)

`localStorage` key `mandelbrot_settings` stores:

| Field | Type | Description |
|---|---|---|
| `colors` | array | Color stop RGBA objects |
| `weights` | array | Gradient segment weights |
| `colorPeriod` | number | LUT cycle length |
| `zoomSpeed` | number | Scroll zoom factor |
| `panSpeed` | number | Arrow key pan fraction |
| `keyZoomSpeed` | number | W/S zoom factor |
| `maxIterAdjustFactor` | number | Reserved |
| `maxiterMode` | string | `'Dynamic'` or `'Fixed'` |
| `baseMaxIter` | number | Base iteration limit |
| `smartphoneMode` | boolean | Whether smartphone mode is active |

Bookmarks are stored separately under `mandelbrot_bookmarks` as a JSON array of `{ name, x, y, viewport }`.

## Arbitrary Precision (`bigfloat.js`)

The center coordinate is tracked as a `BigFloat` (multi-word fixed-point) to avoid precision loss at deep zoom levels. All pan operations add a `BigFloat.fromNumber(delta)` to the current center. The f64 approximation `state.centerX/Y` is derived from `BigFloat.toNumber()` for passing to the GPU.
