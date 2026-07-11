// f16xf16->f32 tensor-core GEMM at Framegen's conv shape, vs tuned f16 subgroup conv (0.53 ms).
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const dawn = require(process.argv[2]);
globalThis.GPUBufferUsage = { MAP_READ:1, MAP_WRITE:2, COPY_SRC:4, COPY_DST:8, INDEX:16, VERTEX:32, UNIFORM:64, STORAGE:128, INDIRECT:256, QUERY_RESOLVE:512 };
globalThis.GPUMapMode = { READ:1, WRITE:2 };
const NACC = Number(process.argv[3] || 4);
const gpu = dawn.create([
  'enable-dawn-features=allow_unsafe_apis,vulkan_enable_f16_on_nvidia',
  'backend=vulkan', 'adapter=NVIDIA',
]);
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter.requestDevice({
  requiredFeatures: ['shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix'],
});
device.addEventListener('uncapturederror', (e) => console.log('DEV ERR:', e.error.message.slice(0, 300)));

const M = 192, K = 1728, N = 1600;
const TM = 16, TN = 16, TK = 16; // f16 16x16x16 -> f32 config

const code = `
enable f16;
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3u) {
  let m0 = wid.y * ${TM}u;
  let n0 = wid.x * ${TN * NACC}u;
  ${Array.from({length: NACC}, (_, i) => `var acc${i}: subgroup_matrix_result<f32, ${TN}, ${TM}>;`).join('\n  ')}
  for (var k = 0u; k < ${K}u; k += ${TK}u) {
    let a = subgroupMatrixLoad<subgroup_matrix_left<f16, ${TK}, ${TM}>>(&A, m0 * ${K}u + k, false, ${K});
    ${Array.from({length: NACC}, (_, i) => `
    let b${i} = subgroupMatrixLoad<subgroup_matrix_right<f16, ${TN}, ${TK}>>(&B, k * ${N}u + n0 + ${TN * i}u, false, ${N});
    acc${i} = subgroupMatrixMultiplyAccumulate(a, b${i}, acc${i});`).join('')}
  }
  ${Array.from({length: NACC}, (_, i) => `subgroupMatrixStore(&C, m0 * ${N}u + n0 + ${TN * i}u, acc${i}, false, ${N});`).join('\n  ')}
}`;

device.pushErrorScope('validation');
const mod = device.createShaderModule({ code });
const ci = await mod.getCompilationInfo();
for (const m of ci.messages) if (m.type === 'error') console.log('COMPILE:', m.message.slice(0, 300));
const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
const perr = await device.popErrorScope();
if (perr) { console.log('PIPE ERR:', perr.message.slice(0, 400)); process.exit(1); }

function f32tof16(v) {
  const f = new Float32Array([v]); const u = new Uint32Array(f.buffer)[0];
  const s = (u >> 16) & 0x8000; let e = ((u >> 23) & 0xff) - 112; const m = u & 0x7fffff;
  if (e <= 0) return s; if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | (m >> 13);
}
function f16tof32(h) {
  const s = (h & 0x8000) << 16; let e = (h >> 10) & 0x1f; let m = h & 0x3ff;
  if (e === 0) { if (m === 0) return s ? -0 : 0; while (!(m & 0x400)) { m <<= 1; e--; } e++; m &= 0x3ff; }
  else if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
  const bits = s | ((e + 112) << 23) | (m << 13);
  return new Float32Array(new Uint32Array([bits]).buffer)[0];
}
const a16 = new Uint16Array(M * K), b16 = new Uint16Array(K * N);
for (let i = 0; i < a16.length; i++) a16[i] = f32tof16(((i * 7 + 3) % 13) / 13);
for (let i = 0; i < b16.length; i++) b16[i] = f32tof16(((i * 5 + 1) % 11) / 11);
const mk = (d) => { const b = device.createBuffer({ size: d.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, d.buffer); return b; };
const bufA = mk(a16), bufB = mk(b16);
const bufC = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufB } },
  { binding: 2, resource: { buffer: bufC } }]});
const gx = N / (TN * NACC), gy = M / TM;
for (let w = 0; w < 3; w++) {
  const enc = device.createCommandEncoder(); const p = enc.beginComputePass();
  p.setPipeline(pipe); p.setBindGroup(0, bind); p.dispatchWorkgroups(gx, gy); p.end();
  device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
}
const reps = 200;
const enc = device.createCommandEncoder(); const p = enc.beginComputePass();
p.setPipeline(pipe); p.setBindGroup(0, bind);
for (let i = 0; i < reps; i++) p.dispatchWorkgroups(gx, gy);
p.end();
const t0 = performance.now();
device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
const ms = (performance.now() - t0) / reps;

// correctness vs CPU (f16 inputs, f32 accumulate)
const read = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(bufC, 0, read, 0, M * N * 4);
device.queue.submit([e2.finish()]); await read.mapAsync(GPUMapMode.READ);
const out = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
const A32 = Float32Array.from(a16, f16tof32), B32 = Float32Array.from(b16, f16tof32);
let maxRelErr = 0;
for (const [m, n] of [[0,0],[7,133],[100,999],[191,1599]]) {
  let s = 0; for (let k = 0; k < K; k++) s += A32[m*K+k] * B32[k*N+n];
  const rel = Math.abs(out[m*N+n] - s) / (Math.abs(s) + 1e-6);
  maxRelErr = Math.max(maxRelErr, rel);
}
const gflop = 2 * M * N * K / 1e9;
console.log(`f16 tensor GEMM (nacc=${NACC}): ${ms.toFixed(4)} ms  (${(gflop/ms).toFixed(1)} TFLOPS)  maxRelErr=${(maxRelErr*100).toFixed(2)}%`);
console.log(`our tuned f16 subgroup conv, same shape: 0.530 ms  ->  speedup ${(0.530/ms).toFixed(2)}x`);
process.stdout.write('', () => process.exit(0));
