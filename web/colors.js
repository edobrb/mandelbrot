/**
 * Color gradient system — replicates the original C# gradient interpolation.
 * Builds a precomputed color lookup table (LUT) indexed by iteration count.
 */

/**
 * Linear interpolation matching the original MandelbrotHelper.linear().
 */
function linear(x, x0, x1, y0, y1) {
    if (x1 - x0 === 0) return (y0 + y1) / 2;
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

/**
 * Pack RGBA channels into a single uint32 matching the original GetRGBA() byte order:
 * (A << 24) | (B << 16) | (G << 8) | R  →  little-endian bytes: R, G, B, A
 */
export function packRGBA(r, g, b, a) {
    return ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

/**
 * Compute a gradient color for a given value, matching MandelbrotHelper.GetLinearGradient().
 *
 * @param {number} v          - Current value (iteration count)
 * @param {number} minV       - Minimum value (0)
 * @param {number} maxV       - Maximum value (maxIter)
 * @param {Array}  colors     - Array of {r, g, b, a} color stops
 * @param {Array}  weights    - Array of segment weights
 * @param {Function} gradientFn - Non-linear mapping function [0,1] → [0,1]
 * @returns {{r: number, g: number, b: number, a: number}}
 */
export function getLinearGradient(v, minV, maxV, colors, weights, gradientFn) {
    const sumWeights = weights.reduce((a, b) => a + b, 0);

    let base;
    if (maxV - minV === 0) {
        base = 0;
    } else {
        base = (v - minV) / (maxV - minV);
    }

    base = gradientFn(base) * sumWeights;

    let i = 0;
    while (i < weights.length - 1 && base > weights[i]) {
        base -= weights[i];
        i++;
    }

    if (i >= weights.length) i = weights.length - 1;
    base = Math.min(base, weights[i]);

    const r = Math.round(Math.max(0, Math.min(255, linear(base, 0, weights[i], colors[i].r, colors[i + 1].r))));
    const g = Math.round(Math.max(0, Math.min(255, linear(base, 0, weights[i], colors[i].g, colors[i + 1].g))));
    const b = Math.round(Math.max(0, Math.min(255, linear(base, 0, weights[i], colors[i].b, colors[i + 1].b))));
    const a = Math.round(Math.max(0, Math.min(255, linear(base, 0, weights[i], colors[i].a, colors[i + 1].a))));

    return { r, g, b, a };
}

/**
 * Build a precomputed color lookup table (Uint32Array) for iteration counts 0..maxIter.
 * The GPU kernel indexes into this table: output_color = color_lut[iteration_count].
 *
 * @param {number}   maxIter    - Maximum iteration count
 * @param {Array}    colors     - Color stop array [{r,g,b,a}, ...]
 * @param {Array}    weights    - Segment weight array
 * @param {Function} gradientFn - Non-linear gradient mapping
 * @returns {Uint32Array} LUT of length maxIter + 1
 */
export function buildColorLUT(maxIter, colors, weights, gradientFn) {
    const lut = new Uint32Array(maxIter + 1);
    for (let i = 0; i <= maxIter; i++) {
        const c = getLinearGradient(i, 0, maxIter, colors, weights, gradientFn);
        lut[i] = packRGBA(c.r, c.g, c.b, c.a);
    }
    return lut;
}

/**
 * Draw the color gradient as a horizontal bar onto a 2D canvas (for preview).
 */
export function drawGradientPreview(canvas, colors, weights, gradientFn) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    for (let x = 0; x < w; x++) {
        const c = getLinearGradient(x, 0, w - 1, colors, weights, gradientFn);
        ctx.fillStyle = `rgb(${c.r},${c.g},${c.b})`;
        ctx.fillRect(x, 0, 1, h);
    }
}
