# Mandelbrot Visualizer — Implementation Documentation

## Architecture

The project is split into two modules:

- **Mandelbrot_Generator**: GPU-accelerated Mandelbrot set computation using CUDAfy.NET (OpenCL backend). Supports multi-GPU workload distribution.
- **Mandelbrot_View**: Real-time viewer built on MonoGame/XNA. Handles input, rendering, settings, and screenshot export.

## Mandelbrot Computation

### Core Algorithm (`MandelbrotCode.cs`)

Standard escape-time algorithm on the GPU. Each thread computes one pixel:

```
z = 0
c = pixel_coordinate_in_complex_plane
while iterations < max_iter AND |z|² < 4:
    z = z² + c
    iterations++
```

The pixel's final iteration count is used as an index into a precomputed color lookup table: `buffer[y * res_x + x] = colors[iteration_count]`.

### GPU Execution Modes

Two modes exist:

1. **Single-pass** (`calculate`): Runs the full iteration loop in one kernel launch. Used when `maxiter_per_step >= max_iter` or `maxiter_per_step <= 0`.
2. **Multi-pass** (`initialize` → `iterate` × N → `finalize`): Splits iterations into chunks of `maxiter_per_step`. State (`z`, iteration count, `c`) is stored in device arrays between passes. Used to avoid GPU timeouts on high iteration counts.

### Multi-GPU (`MandelbrotMultiGPU.cs`)

The Y axis of the image is partitioned across GPUs according to `Devices_PortionY` weights. Each GPU renders its horizontal strip in parallel using `Task`. Results are merged into a shared `uint[]` buffer.

### Thread Mapping

- `blockIdx.x` = row (y coordinate)
- `threadIdx.x` = column within a split (x coordinate offset)
- Thread ID for state arrays: `tid = blockIdx.x + res_y * threadIdx.x`

## Viewer (`Game1.cs`)

Built on MonoGame. The render loop:

1. Processes user input (pan, zoom, maxiter adjustment).
2. If the view changed, triggers a GPU render (on a background thread in `Fluid` mode, synchronously in `Forced` mode, on Enter key in `Manual` mode).
3. Displays the rendered texture, scaling/cropping the cached render to match the current viewport for smooth panning while a new frame computes.

### Navigation

| Key | Action |
|---|---|
| W / S | Zoom in / out |
| Arrow keys | Pan |
| M / N | Increase / decrease max iterations |
| Left click | Center view on click position |
| Enter | Force re-render (Manual mode) |
| R | Reload `settings.json` (colors, gradient, modes) |
| I | Toggle info overlay |
| F11 | Toggle fullscreen |
| F12 | Save PNG screenshot at `ScreenShotResolution` |

### Dynamic Max Iterations

When `MaxiterMode` is `Dynamic`, the iteration limit adapts to zoom level:

```
maxiter = sqrt(2 * sqrt(|1 - sqrt(5 / viewport)|)) * (base_maxiter / 10)
```

This increases detail automatically as the user zooms deeper.

## Color Design

### Pipeline

The color system converts an integer iteration count (0 to `max_iter`) into an RGBA pixel value via gradient interpolation.

### Color Stops and Weights

Defined in `settings.json`:

```
Colors:  DimGray(69,69,69) → DarkGray(169,169,169) → Black(0,0,0) → Red(255,0,0) → DarkRed(139,0,0) → Black(0,0,0)
Weights: [1.0, 1.0, 1.0, 1.0, 1.0]
```

There are 6 color stops and 5 weights. Each weight defines the relative size of the segment between two adjacent stops. Equal weights (all `1.0`) mean each segment spans the same portion of the gradient.

### Gradient Function

A configurable non-linear mapping applied before segment lookup:

```
GradientFunction: "ln(x * 9 + 1) / ln(10)"
```

This is a logarithmic curve. Input `x` is the normalized iteration position `[0, 1]` (i.e., `iteration / max_iter`). The function compresses the gradient toward the beginning — low iteration values (exterior of the set) get more color variation, while high iteration values (near the boundary) are compressed. This avoids a washed-out look at high zoom levels.

### Interpolation (`MandelbrotHelper.cs`)

```
GetLinearGradient(iteration, 0, max_iter, colors[], weights[], gradientFunction):
  1. Normalize: base = (iteration - 0) / (max_iter - 0)    → [0, 1]
  2. Apply gradient function: base = gradientFunction(base) * sum(weights)
  3. Find segment: walk through weights until cumulative sum exceeds base
  4. Linear interpolate RGBA between the two bounding color stops within the segment
```

Each channel (R, G, B, A) is interpolated independently using linear interpolation:

```
channel = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
```

### Precomputed Lookup Table

Colors are NOT computed per-pixel on the GPU. Instead, a `uint[max_iter + 1]` array is built on the CPU and uploaded to GPU memory. The GPU kernel simply does `buffer[pixel] = colors[iteration_count]`. This makes the color system entirely CPU-configurable without recompiling GPU kernels.

### Color Encoding

- For screen display (MonoGame): RGBA byte order → `uint` via `GetRGBA()`
- For PNG screenshots (System.Drawing): ARGB byte order → `uint` via `GetARGB()`

## Settings (`settings.json`)

All configuration is loaded from `settings.json` at startup and can be hot-reloaded with the `R` key:

| Setting | Description |
|---|---|
| `InitialCenterX/Y` | Starting complex plane coordinates |
| `InitialViewportSizeY` | Initial vertical span in the complex plane |
| `InitialMaxIter` | Base iteration limit |
| `ResolutionX/Y` | Window size |
| `RenderResolutionX/Y` | Internal render resolution (can differ from window) |
| `ScreenShotResolutionX/Y` | Screenshot resolution (e.g., 4K) |
| `MaxiterMode` | `Static` or `Dynamic` |
| `MaxIterDynamicFunction` | Math expression for dynamic maxiter |
| `RenderMode` | `Manual`, `Forced`, or `Fluid` |
| `Colors` | Array of RGBA color stops |
| `Weight` | Relative segment sizes between color stops |
| `GradientFunction` | Non-linear mapping expression applied to normalized iteration |
| `Devices_OpenCL_ID` | GPU device IDs |
| `Devices_SplitX` | X-axis thread splits per GPU |
| `Maxiter_Per_Step` | Iterations per kernel pass |
| `Devices_PortionY` | Y-axis workload distribution weights |
