// ============================================================
// mandelbrot.wgsl — Compute shader for Mandelbrot set
// Uses octo-single (OS) arithmetic for ultra-deep zoom.
// OS = 8×f32, ~192-bit mantissa (~56 decimal digits).
// ============================================================

// ----- OS struct -----
// x = x0 + x1 + … + x7,  each |x_{k+1}| ≤ 0.5·ulp(x_k)

struct OS {
    x0: f32, x1: f32, x2: f32, x3: f32,
    x4: f32, x5: f32, x6: f32, x7: f32,
}

// ----- Primitive error-free transforms -----

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    return vec2<f32>(s, b - (s - a));
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    return vec2<f32>(s, (a - (s - v)) + (b - v));
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    return vec2<f32>(p, fma(a, b, -p));
}

// ----- Three-term helpers -----

// Returns (a', b', c') with a'+b'+c' = a+b+c, non-overlapping
fn three_sum(a: f32, b: f32, c: f32) -> vec3<f32> {
    let t1 = two_sum(a, b);
    let t2 = two_sum(c, t1.x);
    let t3 = two_sum(t1.y, t2.y);
    return vec3<f32>(t2.x, t3.x, t3.y);
}

// Like three_sum but keeps only 2 components (sloppy third)
fn three_sum2(a: f32, b: f32, c: f32) -> vec2<f32> {
    let t1 = two_sum(a, b);
    let t2 = two_sum(c, t1.x);
    return vec2<f32>(t2.x, t1.y + t2.y);
}

// ----- Renormalization: 9 → 8 components -----

fn os_renorm(
    c0: f32, c1: f32, c2: f32, c3: f32,
    c4: f32, c5: f32, c6: f32, c7: f32, c8: f32
) -> OS {
    var t: vec2<f32>;

    // Pass 1: bottom-up cascade — accumulate into a single sum,
    // saving the round-off at each merge as b1..b7.
    t = two_sum(c7, c8); var b7 = t.y; var carry = t.x;
    t = two_sum(c6, carry); var b6 = t.y; carry = t.x;
    t = two_sum(c5, carry); var b5 = t.y; carry = t.x;
    t = two_sum(c4, carry); var b4 = t.y; carry = t.x;
    t = two_sum(c3, carry); var b3 = t.y; carry = t.x;
    t = two_sum(c2, carry); var b2 = t.y; carry = t.x;
    t = two_sum(c1, carry); var b1 = t.y; carry = t.x;
    t = two_sum(c0, carry); let a0 = t.x; let a1 = t.y;

    // Pass 2: forward cascade — spread the round-off back down
    // producing a non-overlapping expansion.
    t = two_sum(a0, a1); var s0 = t.x; var e = t.y;
    t = two_sum(e, b1);  var s1 = t.x; e = t.y;
    t = two_sum(e, b2);  var s2 = t.x; e = t.y;
    t = two_sum(e, b3);  var s3 = t.x; e = t.y;
    t = two_sum(e, b4);  var s4 = t.x; e = t.y;
    t = two_sum(e, b5);  var s5 = t.x; e = t.y;
    t = two_sum(e, b6);  var s6 = t.x; e = t.y;
    let s7 = e + b7; // last component: error dropped (level 8)

    return OS(s0, s1, s2, s3, s4, s5, s6, s7);
}

// ----- OS Addition -----

fn os_add(a: OS, b: OS) -> OS {
    var r: vec2<f32>;
    // Pairwise sums + carry tracking
    r = two_sum(a.x0, b.x0); var s0 = r.x; var t0 = r.y;
    r = two_sum(a.x1, b.x1); var s1 = r.x; var t1 = r.y;
    r = two_sum(a.x2, b.x2); var s2 = r.x; var t2 = r.y;
    r = two_sum(a.x3, b.x3); var s3 = r.x; var t3 = r.y;
    r = two_sum(a.x4, b.x4); var s4 = r.x; var t4 = r.y;
    r = two_sum(a.x5, b.x5); var s5 = r.x; var t5 = r.y;
    r = two_sum(a.x6, b.x6); var s6 = r.x; var t6 = r.y;
    var s7 = a.x7 + b.x7; // sloppy last component

    // Merge carry t0 and secondary errors t1..t6 level by level
    r = two_sum(s1, t0); s1 = r.x; t0 = r.y;

    var v: vec3<f32>;
    v = three_sum(s2, t0, t1); s2 = v.x; t0 = v.y; t1 = v.z;
    v = three_sum(s3, t0, t2); s3 = v.x; t0 = v.y; t2 = v.z;
    v = three_sum(s4, t0, t3); s4 = v.x; t0 = v.y; t3 = v.z;
    v = three_sum(s5, t0, t4); s5 = v.x; t0 = v.y; t4 = v.z;
    v = three_sum(s6, t0, t5); s6 = v.x; t0 = v.y; t5 = v.z;

    let v2 = three_sum2(s7, t0, t6); s7 = v2.x; t0 = v2.y;

    // Remaining tiny errors → last renorm slot
    let t8 = t0 + t1 + t2 + t3 + t4 + t5;

    return os_renorm(s0, s1, s2, s3, s4, s5, s6, s7, t8);
}

fn os_neg(a: OS) -> OS {
    return OS(-a.x0, -a.x1, -a.x2, -a.x3, -a.x4, -a.x5, -a.x6, -a.x7);
}

fn os_sub(a: OS, b: OS) -> OS {
    return os_add(a, os_neg(b));
}

// Exact doubling (×2 is exact in f32 as long as no overflow)
fn os_double(a: OS) -> OS {
    return OS(a.x0*2.0, a.x1*2.0, a.x2*2.0, a.x3*2.0,
              a.x4*2.0, a.x5*2.0, a.x6*2.0, a.x7*2.0);
}

// ----- Scalar × OS multiplication -----

fn f32_os_mul(s: f32, b: OS) -> OS {
    // s·b = Σ s·b.xi = Σ (pi.x + pi.y) exactly
    let p0 = two_prod(s, b.x0);
    let p1 = two_prod(s, b.x1);
    let p2 = two_prod(s, b.x2);
    let p3 = two_prod(s, b.x3);
    let p4 = two_prod(s, b.x4);
    let p5 = two_prod(s, b.x5);
    let p6 = two_prod(s, b.x6);
    let sl7 = s * b.x7 + p6.y; // sloppy level 7

    // Merge adjacent pairs (pi.y , p_{i+1}.x) at level i+1
    var t: vec2<f32>;
    t = two_sum(p0.y, p1.x); let v1 = t.x; let e1 = t.y;
    t = two_sum(p1.y, p2.x); let v2 = t.x; let e2 = t.y;
    t = two_sum(p2.y, p3.x); let v3 = t.x; let e3 = t.y;
    t = two_sum(p3.y, p4.x); let v4 = t.x; let e4 = t.y;
    t = two_sum(p4.y, p5.x); let v5 = t.x; let e5 = t.y;
    t = two_sum(p5.y, p6.x); let v6 = t.x; let e6 = t.y;

    // Split into main values and sub-ULP corrections, then add
    let main = os_renorm(p0.x, v1, v2, v3, v4, v5, v6, sl7, 0.0);
    let corr = os_renorm(e1, e2, e3, e4, e5, e6, 0.0, 0.0, 0.0);
    return os_add(main, corr);
}

// ----- OS × OS multiplication -----

fn os_mul(a: OS, b: OS) -> OS {
    // a·b = Σ a.xi · b  (sum scalar×OS products from largest to smallest)
    var acc = f32_os_mul(a.x0, b);
    acc = os_add(acc, f32_os_mul(a.x1, b));
    acc = os_add(acc, f32_os_mul(a.x2, b));
    acc = os_add(acc, f32_os_mul(a.x3, b));
    acc = os_add(acc, f32_os_mul(a.x4, b));
    acc = os_add(acc, f32_os_mul(a.x5, b));
    acc = os_add(acc, f32_os_mul(a.x6, b));
    acc = os_add(acc, f32_os_mul(a.x7, b));
    return acc;
}

// ----- Mandelbrot Parameters -----

struct Params {
    x0_0: f32, x0_1: f32, x0_2: f32, x0_3: f32,
    x0_4: f32, x0_5: f32, x0_6: f32, x0_7: f32,
    x1_0: f32, x1_1: f32, x1_2: f32, x1_3: f32,
    x1_4: f32, x1_5: f32, x1_6: f32, x1_7: f32,
    y0_0: f32, y0_1: f32, y0_2: f32, y0_3: f32,
    y0_4: f32, y0_5: f32, y0_6: f32, y0_7: f32,
    y1_0: f32, y1_1: f32, y1_2: f32, y1_3: f32,
    y1_4: f32, y1_5: f32, y1_6: f32, y1_7: f32,
    res_x   : u32,
    res_y   : u32,
    max_iter: u32,
    _pad    : u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> color_lut : array<u32>;
@group(0) @binding(2) var<storage, read_write> output : array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let px = gid.x;
    let py = gid.y;

    if (px >= params.res_x || py >= params.res_y) { return; }

    // Viewport corners as OS values
    let x0 = OS(params.x0_0, params.x0_1, params.x0_2, params.x0_3,
                params.x0_4, params.x0_5, params.x0_6, params.x0_7);
    let x1 = OS(params.x1_0, params.x1_1, params.x1_2, params.x1_3,
                params.x1_4, params.x1_5, params.x1_6, params.x1_7);
    let y0 = OS(params.y0_0, params.y0_1, params.y0_2, params.y0_3,
                params.y0_4, params.y0_5, params.y0_6, params.y0_7);
    let y1 = OS(params.y1_0, params.y1_1, params.y1_2, params.y1_3,
                params.y1_4, params.y1_5, params.y1_6, params.y1_7);

    // c = corner + fraction × span  (OS precision)
    let dx = os_sub(x1, x0);
    let dy = os_sub(y1, y0);
    let fx = OS(f32(px) / f32(params.res_x), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let fy = OS(f32(py) / f32(params.res_y), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let cx = os_add(x0, os_mul(dx, fx));
    let cy = os_add(y0, os_mul(dy, fy));

    // z = 0,  iterate z = z² + c,  bail when |z|² ≥ 4
    var zx = OS(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var zy = OS(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var iter = 0u;

    loop {
        if (iter >= params.max_iter) { break; }

        let zx2  = os_mul(zx, zx);
        let zy2  = os_mul(zy, zy);
        let mag2 = os_add(zx2, zy2);

        if ((mag2.x0 + mag2.x1) >= 4.0) { break; }

        // new_zy = 2·zx·zy + cy  (use os_double to avoid a full multiply)
        let zxzy   = os_mul(zx, zy);
        let new_zy = os_add(os_double(zxzy), cy);
        // new_zx = zx² - zy² + cx
        let new_zx = os_add(os_sub(zx2, zy2), cx);

        zx = new_zx;
        zy = new_zy;
        iter = iter + 1u;
    }

    output[py * params.res_x + px] = color_lut[iter];
}
