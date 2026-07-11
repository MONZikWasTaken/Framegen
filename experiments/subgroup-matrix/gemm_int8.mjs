// single-variant bench (one pipeline per process - the March dawn.node is fragile)
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const NACC = Number(process.argv[2] || 1);
const gpu = create(['backend=vulkan', 'enable-dawn-features=allow_unsafe_apis']);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: ['subgroups', 'chromium-experimental-subgroup-matrix'],
});
const M = 192, K = 1728, N = 1600;
const code = `
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3u) {
  let m0 = wid.y * 16u;
  let n0 = wid.x * ${16 * NACC}u;
  ${Array.from({length: NACC}, (_, i) => `var acc${i}: subgroup_matrix_result<u32, 16, 16>;`).join('\n  ')}
  for (var k = 0u; k < ${K}u; k += 32u) {
    let a = subgroupMatrixLoad<subgroup_matrix_left<u8, 32, 16>>(&A, m0 * ${K}u + k, false, ${K});
    ${Array.from({length: NACC}, (_, i) => `
    let b${i} = subgroupMatrixLoad<subgroup_matrix_right<u8, 16, 32>>(&B, k * ${N}u + n0 + ${16 * i}u, false, ${N});
    acc${i} = subgroupMatrixMultiplyAccumulate(a, b${i}, acc${i});`).join('')}
  }
  ${Array.from({length: NACC}, (_, i) => `subgroupMatrixStore(&C, m0 * ${N}u + n0 + ${16 * i}u, acc${i}, false, ${N});`).join('\n  ')}
}`;
const a8 = new Uint8Array(M * K);
const b8 = new Uint8Array(K * N);
for (let i = 0; i < a8.length; i++) a8[i] = (i * 7 + 3) & 15;
for (let i = 0; i < b8.length; i++) b8[i] = (i * 5 + 1) & 15;
const mk = (d) => { const b = device.createBuffer({ size: d.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, d.buffer); return b; };
const bufA = mk(a8), bufB = mk(b8);
const bufC = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const pipe = device.createComputePipeline({ layout: 'auto',
  compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufB } },
  { binding: 2, resource: { buffer: bufC } }]});
const gx = N / (16 * NACC), gy = M / 16;
for (let w = 0; w < 2; w++) {
  const enc = device.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(pipe); p.setBindGroup(0, bind); p.dispatchWorkgroups(gx, gy);
  p.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const enc = device.createCommandEncoder();
const p = enc.beginComputePass();
p.setPipeline(pipe); p.setBindGroup(0, bind);
for (let i = 0; i < 100; i++) p.dispatchWorkgroups(gx, gy);
p.end();
const t0 = performance.now();
device.queue.submit([enc.finish()]);
await device.queue.onSubmittedWorkDone();
const ms = (performance.now() - t0) / 100;
// correctness spot check
const read = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const e2 = device.createCommandEncoder();
e2.copyBufferToBuffer(bufC, 0, read, 0, M * N * 4);
device.queue.submit([e2.finish()]);
await read.mapAsync(GPUMapMode.READ);
const out = new Uint32Array(read.getMappedRange().slice(0));
read.unmap();
const refCell = (m, n) => { let s = 0; for (let k = 0; k < K; k++) s += a8[m * K + k] * b8[k * N + n]; return s; };
let ok = true;
for (const [m, n] of [[0, 0], [7, 133], [100, 999], [191, 1599]]) if (out[m * N + n] !== refCell(m, n)) ok = false;
const gmac = M * N * K / 1e9;
console.log(`nacc=${NACC}: ${ms.toFixed(3)} ms  (${(2 * gmac / ms).toFixed(1)} TOPS int8)  correctness=${ok ? 'PASS' : 'FAIL'}`);
process.stdout.write('', () => { process.exit(0); });
