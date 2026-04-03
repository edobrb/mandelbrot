// ============================================================
// mandelbrot_perturb.wgsl — Perturbation theory compute shader
// Uses Zhuoran's rebasing algorithm for glitch prevention.
// Only ONE reference orbit (computed on CPU at arbitrary precision)
// is needed; all pixels compute lightweight f32 delta orbits.
// ============================================================

struct Params {
    half_w       : f32,   // res_x / 2.0
    half_h       : f32,   // res_y / 2.0
    scale_re     : f32,   // viewportSizeX / res_x  (per-pixel real scale)
    scale_im     : f32,   // viewportSizeY / res_y  (per-pixel imag scale)
    res_x        : u32,
    res_y        : u32,
    max_iter     : u32,
    ref_len      : u32,   // length of reference orbit (iterations before it escapes)
    color_period : u32,   // LUT cycle length; escaped pixels use iter % color_period
    _pad1        : u32,
    _pad2        : u32,
    _pad3        : u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> ref_re : array<f32>;
@group(0) @binding(2) var<storage, read> ref_im : array<f32>;
@group(0) @binding(3) var<storage, read> color_lut : array<u32>;
@group(0) @binding(4) var<storage, read_write> output : array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.res_x || py >= params.res_y) {
        return;
    }

    // dc = complex offset of this pixel from the reference (center) point
    let dc_re = (f32(px) - params.half_w + 0.5) * params.scale_re;
    let dc_im = (f32(py) - params.half_h + 0.5) * params.scale_im;

    // Delta orbit
    var dz_re = 0.0f;
    var dz_im = 0.0f;
    var iter = 0u;
    var ref_iter = 0u;

    let bailout_sq = 4.0f;

    loop {
        if (iter >= params.max_iter) {
            break;
        }

        // Perturbation iteration: dz' = 2·dz·Z[ref] + dz² + dc
        let Zr = ref_re[ref_iter];
        let Zi = ref_im[ref_iter];

        // 2·dz·Z = (2·dz_re·Zr − 2·dz_im·Zi, 2·dz_re·Zi + 2·dz_im·Zr)
        // dz²    = (dz_re² − dz_im², 2·dz_re·dz_im)
        let two_dz_re = 2.0 * dz_re;
        let two_dz_im = 2.0 * dz_im;

        let new_dz_re = fma(two_dz_re, Zr, fma(-two_dz_im, Zi, fma(dz_re, dz_re, -dz_im * dz_im + dc_re)));
        let new_dz_im = fma(two_dz_re, Zi, fma(two_dz_im, Zr, 2.0 * dz_re * dz_im + dc_im));

        dz_re = new_dz_re;
        dz_im = new_dz_im;
        ref_iter = ref_iter + 1u;

        // z = Z[ref_iter] + dz  (full complex position)
        let z_re = ref_re[ref_iter] + dz_re;
        let z_im = ref_im[ref_iter] + dz_im;

        let z_mag2 = z_re * z_re + z_im * z_im;

        // Escape check
        if (z_mag2 > bailout_sq) {
            break;
        }

        // Zhuoran's rebasing: if |z| < |dz| or reference orbit exhausted,
        // reset dz = z and restart reference from iteration 0.
        let dz_mag2 = dz_re * dz_re + dz_im * dz_im;
        if (z_mag2 < dz_mag2 || ref_iter >= params.ref_len) {
            dz_re = z_re;
            dz_im = z_im;
            ref_iter = 0u;
        }

        iter = iter + 1u;
    }

    let idx = py * params.res_x + px;
    if (iter >= params.max_iter) {
        output[idx] = color_lut[params.color_period]; // inside color (last LUT slot)
    } else {
        output[idx] = color_lut[iter % params.color_period];
    }
}
