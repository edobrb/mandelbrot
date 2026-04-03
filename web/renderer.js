/**
 * renderer.js — WebGPU setup, pipelines, buffer management, render dispatch.
 *
 * Uses perturbation theory: a single reference orbit (computed on CPU at
 * arbitrary precision) is uploaded to the GPU, and each pixel computes
 * only its lightweight f32 delta orbit.
 */

// ---------- Render shader (inline WGSL) ----------

const RENDER_SHADER = /* wgsl */`
struct RenderParams {
    width  : u32,
    height : u32,
    _pad0  : u32,
    _pad1  : u32,
};

@group(0) @binding(0) var<uniform> rp : RenderParams;
@group(0) @binding(1) var<storage, read> pixels : array<u32>;

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);

    if (x >= rp.width || y >= rp.height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let idx = y * rp.width + x;
    let packed = pixels[idx];

    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u)  & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;

    return vec4<f32>(r, g, b, a);
}
`;

// ---------- Renderer class ----------

export class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;
        this.format = null;

        // Pipelines
        this.computePipeline = null;
        this.renderPipeline = null;

        // Buffers
        this.computeParamsBuffer = null;
        this.refOrbitReBuffer = null;
        this.refOrbitImBuffer = null;
        this.colorLUTBuffer = null;
        this.pixelBuffer = null;
        this.renderParamsBuffer = null;

        // Bind groups
        this.computeBindGroup = null;
        this.renderBindGroup = null;
        this.computeBindGroupLayout = null;
        this.renderBindGroupLayout = null;

        this.width = 0;
        this.height = 0;
        this.currentMaxIter = 0;
        this.currentRefOrbitCapacity = 0;
    }

    async init() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('WebGPU adapter not available.');

        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            },
        });

        this.device.lost.then((info) => {
            console.error('WebGPU device lost:', info.message);
        });

        this.context = this.canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque',
        });

        await this._createPipelines();
        this._createStaticBuffers();
    }

    async _createPipelines() {
        // ----- Perturbation compute pipeline -----
        const computeShaderSrc = await fetch('mandelbrot_perturb.wgsl').then((r) => r.text());
        const computeModule = this.device.createShaderModule({ code: computeShaderSrc });

        this.computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        // ----- Render pipeline (same as before) -----
        const renderModule = this.device.createShaderModule({ code: RENDER_SHADER });

        this.renderBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
            ],
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.renderBindGroupLayout] }),
            vertex: { module: renderModule, entryPoint: 'vs' },
            fragment: {
                module: renderModule,
                entryPoint: 'fs',
                targets: [{ format: this.format }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    _createStaticBuffers() {
        // Compute params uniform (48 bytes: 4 f32 + 8 u32)
        this.computeParamsBuffer = this.device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Render params uniform (16 bytes)
        this.renderParamsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /**
     * Resize internal buffers to match new canvas pixel dimensions.
     */
    resize(width, height) {
        if (width === this.width && height === this.height) return;
        this.width = width;
        this.height = height;

        this.canvas.width = width;
        this.canvas.height = height;

        // Recreate pixel output buffer
        if (this.pixelBuffer) this.pixelBuffer.destroy();
        this.pixelBuffer = this.device.createBuffer({
            size: width * height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Update render params
        const rp = new Uint32Array([width, height, 0, 0]);
        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, rp);

        this._rebuildBindGroups();
    }

    /**
     * Upload a new color LUT to the GPU.
     */
    updateColorLUT(lut, maxIter) {
        if (this.colorLUTBuffer) this.colorLUTBuffer.destroy();
        this.colorLUTBuffer = this.device.createBuffer({
            size: lut.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.colorLUTBuffer, 0, lut);
        this.currentMaxIter = maxIter;

        this._rebuildBindGroups();
    }

    /**
     * Upload the reference orbit to the GPU.
     * @param {Float32Array} re - Real parts of reference orbit
     * @param {Float32Array} im - Imaginary parts of reference orbit
     * @param {number} length - Number of valid entries
     */
    updateRefOrbit(re, im, length) {
        const byteSize = (length + 1) * 4; // +1 for safety
        const needRealloc = byteSize > this.currentRefOrbitCapacity;

        if (needRealloc) {
            if (this.refOrbitReBuffer) this.refOrbitReBuffer.destroy();
            if (this.refOrbitImBuffer) this.refOrbitImBuffer.destroy();

            this.refOrbitReBuffer = this.device.createBuffer({
                size: byteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.refOrbitImBuffer = this.device.createBuffer({
                size: byteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.currentRefOrbitCapacity = byteSize;
        }

        this.device.queue.writeBuffer(this.refOrbitReBuffer, 0, re, 0, length + 1);
        this.device.queue.writeBuffer(this.refOrbitImBuffer, 0, im, 0, length + 1);

        if (needRealloc) this._rebuildBindGroups();
    }

    _rebuildBindGroups() {
        if (!this.pixelBuffer || !this.colorLUTBuffer || !this.refOrbitReBuffer || !this.refOrbitImBuffer) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.computeParamsBuffer } },
                { binding: 1, resource: { buffer: this.refOrbitReBuffer } },
                { binding: 2, resource: { buffer: this.refOrbitImBuffer } },
                { binding: 3, resource: { buffer: this.colorLUTBuffer } },
                { binding: 4, resource: { buffer: this.pixelBuffer } },
            ],
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.renderParamsBuffer } },
                { binding: 1, resource: { buffer: this.pixelBuffer } },
            ],
        });
    }

    /**
     * Render a frame using perturbation theory.
     *
     * @param {number} viewportSizeX - Width of viewport in complex plane units
     * @param {number} viewportSizeY - Height of viewport in complex plane units
     * @param {number} refLength - Length of the reference orbit
     * @param {number} maxIter - Maximum iterations
     */
    render(viewportSizeX, viewportSizeY, refLength, maxIter, colorPeriod) {
        if (this.width === 0 || this.height === 0) return;
        if (!this.computeBindGroup || !this.renderBindGroup) return;

        // Write compute params
        const buf = new ArrayBuffer(48);
        const f = new Float32Array(buf, 0, 4);
        const u = new Uint32Array(buf, 16, 8);

        f[0] = this.width / 2;             // half_w
        f[1] = this.height / 2;            // half_h
        f[2] = Math.fround(viewportSizeX / this.width);   // scale_re
        f[3] = Math.fround(viewportSizeY / this.height);  // scale_im
        u[0] = this.width;                 // res_x
        u[1] = this.height;                // res_y
        u[2] = maxIter;                    // max_iter
        u[3] = refLength;                  // ref_len
        u[4] = colorPeriod;                // color_period

        this.device.queue.writeBuffer(this.computeParamsBuffer, 0, buf);

        const encoder = this.device.createCommandEncoder();

        // Compute pass
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(this.width / 16),
            Math.ceil(this.height / 16),
        );
        computePass.end();

        // Render pass
        const textureView = this.context.getCurrentTexture().createView();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    /**
     * Read the current pixel buffer from the GPU and return a PNG Blob.
     * Reads pixelBuffer directly — works regardless of canvas swap state.
     * Pixel format is packed uint32: bytes R, G, B, A (little-endian), matching ImageData.
     */
    async captureScreenshot() {
        const w = this.width, h = this.height;
        if (!this.pixelBuffer || w === 0 || h === 0) return null;

        const readback = this.device.createBuffer({
            size: w * h * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.pixelBuffer, 0, readback, 0, w * h * 4);
        this.device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const pixels = new Uint8ClampedArray(readback.getMappedRange().slice(0));
        readback.unmap();
        readback.destroy();

        const offscreen = new OffscreenCanvas(w, h);
        const ctx = offscreen.getContext('2d');
        ctx.putImageData(new ImageData(pixels, w, h), 0, 0);
        return offscreen.convertToBlob({ type: 'image/png' });
    }

    destroy() {
        if (this.computeParamsBuffer) this.computeParamsBuffer.destroy();
        if (this.renderParamsBuffer) this.renderParamsBuffer.destroy();
        if (this.colorLUTBuffer) this.colorLUTBuffer.destroy();
        if (this.pixelBuffer) this.pixelBuffer.destroy();
        if (this.refOrbitReBuffer) this.refOrbitReBuffer.destroy();
        if (this.refOrbitImBuffer) this.refOrbitImBuffer.destroy();
        this.device?.destroy();
    }
}
