/**
 * renderer.js — WebGPU setup, pipelines, buffer management, render dispatch.
 *
 * Compute shader writes u32-packed RGBA pixels to a storage buffer.
 * A fullscreen-triangle render pass unpacks and displays them.
 */

import { buildColorLUT } from './colors.js';

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
    // Fullscreen triangle covering entire clip space
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

    // Unpack ABGR u32 → RGBA float channels
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u)  & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;

    return vec4<f32>(r, g, b, a);
}
`;

// ---------- Helpers ----------

function splitOcto(val) {
    const x0 = Math.fround(val);
    const r1 = val - x0;
    const x1 = Math.fround(r1);
    const r2 = r1 - x1;
    const x2 = Math.fround(r2);
    const r3 = r2 - x2;
    const x3 = Math.fround(r3);
    const r4 = r3 - x3;
    const x4 = Math.fround(r4);
    // JS f64 has 52-bit mantissa; 3 f32 (24-bit each) exhaust it,
    // so x3 onward will be 0. Kept for completeness / future bignum inputs.
    return [x0, x1, x2, x3, x4, 0, 0, 0];
}

// ---------- Renderer class ----------

export class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;
        this.format = null;
        this.computePipeline = null;
        this.renderPipeline = null;

        this.computeParamsBuffer = null;
        this.colorLUTBuffer = null;
        this.pixelBuffer = null;
        this.renderParamsBuffer = null;

        this.computeBindGroup = null;
        this.renderBindGroup = null;

        this.computeBindGroupLayout = null;
        this.renderBindGroupLayout = null;

        this.width = 0;
        this.height = 0;
        this.currentMaxIter = 0;
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
        // ----- Compute pipeline -----
        const computeShaderSrc = await fetch('mandelbrot.wgsl').then((r) => r.text());
        const computeModule = this.device.createShaderModule({ code: computeShaderSrc });

        this.computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.computeBindGroupLayout] }),
            compute: { module: computeModule, entryPoint: 'main' },
        });

        // ----- Render pipeline -----
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
        // Compute params uniform (144 bytes: 32 f32 coords + 4 u32)
        this.computeParamsBuffer = this.device.createBuffer({
            size: 144,
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
        const pixelCount = width * height;
        this.pixelBuffer = this.device.createBuffer({
            size: pixelCount * 4,
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

    _rebuildBindGroups() {
        if (!this.pixelBuffer || !this.colorLUTBuffer) return;

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.computeParamsBuffer } },
                { binding: 1, resource: { buffer: this.colorLUTBuffer } },
                { binding: 2, resource: { buffer: this.pixelBuffer } },
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
     * Render a frame: dispatch compute shader, then draw fullscreen quad.
     *
     * @param {number} x0 - Left edge in complex plane
     * @param {number} x1 - Right edge
     * @param {number} y0 - Top edge
     * @param {number} y1 - Bottom edge
     * @param {number} maxIter - Maximum iterations
     */
    render(x0, x1, y0, y1, maxIter) {
        if (this.width === 0 || this.height === 0) return;
        if (!this.computeBindGroup || !this.renderBindGroup) return;

        // Write compute params (144 bytes: 32 f32 coords + 4 u32)
        const paramsF = new Float32Array([
            ...splitOcto(x0),
            ...splitOcto(x1),
            ...splitOcto(y0),
            ...splitOcto(y1),
        ]);
        const paramsU = new Uint32Array([this.width, this.height, maxIter, 0]);
        const paramsBuf = new ArrayBuffer(144);
        new Float32Array(paramsBuf, 0, 32).set(paramsF);
        new Uint32Array(paramsBuf, 128, 4).set(paramsU);
        this.device.queue.writeBuffer(this.computeParamsBuffer, 0, paramsBuf);

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

    destroy() {
        if (this.computeParamsBuffer) this.computeParamsBuffer.destroy();
        if (this.renderParamsBuffer) this.renderParamsBuffer.destroy();
        if (this.colorLUTBuffer) this.colorLUTBuffer.destroy();
        if (this.pixelBuffer) this.pixelBuffer.destroy();
        this.device?.destroy();
    }
}
