// ============================================================
// mandelbrot.wgsl — Compute shader for Mandelbrot set
// Uses double-single (DS) arithmetic for deep-zoom precision
// ============================================================

// ----- Double-Single Arithmetic -----
// A DS number is vec2<f32>(hi, lo) representing the value hi + lo.
// This gives ~48 bits of mantissa vs ~24 for plain f32.

fn ds_quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let e = b - (s - a);
    return vec2<f32>(s, e);
}

fn ds_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2<f32>(s, e);
}

fn ds_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = ds_two_sum(a.x, b.x);
    let e = a.y + b.y + s.y;
    return ds_quick_two_sum(s.x, e);
}

fn ds_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return ds_add(a, vec2<f32>(-b.x, -b.y));
}

fn ds_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = ds_two_prod(a.x, b.x);
    let e = a.x * b.y + a.y * b.x + p.y;
    return ds_quick_two_sum(p.x, e);
}

// ----- Mandelbrot Parameters -----

struct Params {
    x0_hi : f32,
    x0_lo : f32,
    x1_hi : f32,
    x1_lo : f32,
    y0_hi : f32,
    y0_lo : f32,
    y1_hi : f32,
    y1_lo : f32,
    res_x : u32,
    res_y : u32,
    max_iter : u32,
    _pad : u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> color_lut : array<u32>;
@group(0) @binding(2) var<storage, read_write> output : array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    if (px >= params.res_x || py >= params.res_y) {
        return;
    }

    // Viewport corners as DS values
    let x0 = vec2<f32>(params.x0_hi, params.x0_lo);
    let x1 = vec2<f32>(params.x1_hi, params.x1_lo);
    let y0 = vec2<f32>(params.y0_hi, params.y0_lo);
    let y1 = vec2<f32>(params.y1_hi, params.y1_lo);

    // c = corner + fraction * span  (DS precision)
    let dx = ds_sub(x1, x0);
    let dy = ds_sub(y1, y0);
    let fx = vec2<f32>(f32(px) / f32(params.res_x), 0.0);
    let fy = vec2<f32>(f32(py) / f32(params.res_y), 0.0);
    let cx = ds_add(x0, ds_mul(dx, fx));
    let cy = ds_add(y0, ds_mul(dy, fy));

    // z = 0,  iterate z = z² + c,  bail when |z|² ≥ 4
    var zx = vec2<f32>(0.0, 0.0);
    var zy = vec2<f32>(0.0, 0.0);
    var iter = 0u;

    loop {
        if (iter >= params.max_iter) {
            break;
        }

        let zx2 = ds_mul(zx, zx);
        let zy2 = ds_mul(zy, zy);
        let mag2 = ds_add(zx2, zy2);

        if ((mag2.x + mag2.y) >= 4.0) {
            break;
        }

        // new_zy = 2·zx·zy + cy
        let two_zx = ds_add(zx, zx);
        let new_zy = ds_add(ds_mul(two_zx, zy), cy);
        // new_zx = zx² - zy² + cx
        let new_zx = ds_add(ds_sub(zx2, zy2), cx);

        zx = new_zx;
        zy = new_zy;
        iter = iter + 1u;
    }

    let idx = py * params.res_x + px;
    output[idx] = color_lut[iter];
}
