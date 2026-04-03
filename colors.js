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
export function buildColorLUT(colorPeriod, colors, weights, gradientFn) {
    // colorPeriod entries for the cycle, +1 for the inside (non-escaped) color
    const lut = new Uint32Array(colorPeriod + 1);
    for (let i = 0; i < colorPeriod; i++) {
        const c = getLinearGradient(i, 0, colorPeriod - 1, colors, weights, gradientFn);
        lut[i] = packRGBA(c.r, c.g, c.b, c.a);
    }
    // Inside color: use the last color stop from the palette
    const inside = colors[colors.length - 1];
    lut[colorPeriod] = packRGBA(inside.r, inside.g, inside.b, inside.a);
    return lut;
}

/**
 * Generate a random aesthetically-pleasing color palette.
 *
 * Rules that make Mandelbrot palettes look good:
 *  - Start and end dark (continuity when the cycle wraps)
 *  - Harmonious hue relationships (analogous / triadic / split-complementary)
 *  - Vivid, saturated midtones — muted stops kill contrast
 *  - 4-6 stops: enough variety, not noisy
 *
 * Returns { colors, weights, colorPeriod }.
 */
export function randomPalette() {
    const rand  = (a, b) => a + Math.random() * (b - a);
    const pick  = (arr) => arr[Math.floor(Math.random() * arr.length)];

    // Convert HSL (h:0-360, s/l:0-1) to {r,g,b,a}
    function hsl(h, s, l) {
        h = ((h % 360) + 360) % 360 / 360;
        s = Math.max(0, Math.min(1, s));
        l = Math.max(0, Math.min(1, l));
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs((h * 6) % 2 - 1));
        const m = l - c / 2;
        let r, g, b;
        const h6 = h * 6;
        if      (h6 < 1) { r = c; g = x; b = 0; }
        else if (h6 < 2) { r = x; g = c; b = 0; }
        else if (h6 < 3) { r = 0; g = c; b = x; }
        else if (h6 < 4) { r = 0; g = x; b = c; }
        else if (h6 < 5) { r = x; g = 0; b = c; }
        else             { r = c; g = 0; b = x; }
        return { r: Math.round((r + m) * 255), g: Math.round((g + m) * 255), b: Math.round((b + m) * 255), a: 255 };
    }

    const baseHue = rand(0, 360);

    // Hue relationships for the mid-stop colors
    const keyHues = pick([
        [baseHue, baseHue + 30, baseHue + 60],                    // analogous
        [baseHue, baseHue + 120, baseHue + 240],                   // triadic
        [baseHue, baseHue + 150, baseHue + 210],                   // split-complementary
        [baseHue, baseHue + 180],                                  // complementary
        [baseHue, baseHue + 30, baseHue + 180, baseHue + 210],     // double complementary
    ]);

    const n = pick([4, 4, 5, 5, 6]); // weighted toward 4-5
    const colors = [];

    // First stop — very dark, slight tint of base hue
    colors.push(hsl(baseHue, rand(0.2, 0.6), rand(0.03, 0.09)));

    // Middle stops — vivid, using the key hues
    for (let i = 1; i < n - 1; i++) {
        const hue = keyHues[i % keyHues.length] + rand(-12, 12);
        const sat = rand(0.70, 1.00);
        const lit = rand(0.28, 0.62);
        colors.push(hsl(hue, sat, lit));
    }

    // Last stop — dark again so the cycle wraps smoothly
    colors.push(hsl(keyHues[keyHues.length - 1], rand(0.1, 0.5), rand(0.03, 0.09)));

    const weights    = Array(n - 1).fill(1.0);
    const colorPeriod = pick([96, 128, 192, 256, 384, 512]);

    return { colors, weights, colorPeriod };
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
