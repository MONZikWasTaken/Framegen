// INT8 tensor-core GEMM via chromium_experimental_subgroup_matrix.
// Stage 1: correctness on one tile vs CPU. Stage 2: timing at the conv shape.
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);

const gpu = create(['backend=vulkan', 'enable-dawn-features=allow_unsafe_apis']);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: ['subgroups', 'chromium-experimental-subgroup-matrix', 'timestamp-query'],
});
device.addEventListener('uncapturederror', (e) => console.log('DEVICE ERR:', e.error.message.slice(0, 700)));

const M = 16, N = 16, K = 32;

// dims order guess #1: subgroup_matrix_left<T, cols, rows> = <u8, K, M>
const code = /* wgsl */`
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<u32>;
@compute @workgroup_size(32)
fn main() {
  let a = subgroupMatrixLoad<subgroup_matrix_left<u8, ${K}, ${M}>>(&A, 0, false, ${K});
  let b = subgroupMatrixLoad<subgroup_matrix_right<u8, ${N}, ${K}>>(&B, 0, false, ${N});
  var acc: subgroup_matrix_result<u32, ${N}, ${M}>;
  acc = subgroupMatrixMultiplyAccumulate(a, b, acc);
  subgroupMatrixStore(&C, 0, acc, false, ${N});
}`;
device.pushErrorScope('validation');
const mod = device.createShaderModule({ code });
const ci = await mod.getCompilationInfo();
for (const m of ci.messages) if (m.type === 'error') console.log('COMPILE ERR:', m.message.slice(0, 700));
const verr = await device.popErrorScope();
if (verr) { console.log('MODULE ERR:', verr.message.slice(0, 700)); process.exit(1); }

// data: A[m][k] = (m+k) % 7, B[k][n] = (k*2+n) % 5
const a8 = new Uint8Array(M * K);
const b8 = new Uint8Array(K * N);
for (let m = 0; m < M; m++) for (let k = 0; k < K; k++) a8[m * K + k] = (m + k) % 7;
for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) b8[k * N + n] = (k * 2 + n) % 5;
const ref = new Uint32Array(M * N);
for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
  let s = 0;
  for (let k = 0; k < K; k++) s += a8[m * K + k] * b8[k * N + n];
  ref[m * N + n] = s;
}
const mkBuf = (data, usage) => {
  const buf = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4, usage });
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
};
const bufA = mkBuf(a8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
const bufB = mkBuf(b8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
const bufC = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const read = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

device.pushErrorScope('validation');
const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
const perr = await device.popErrorScope();
if (perr) { console.log('PIPELINE ERR:', perr.message.slice(0, 700)); process.exit(1); }

const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: bufA } },
  { binding: 1, resource: { buffer: bufB } },
  { binding: 2, resource: { buffer: bufC } },
]});
const enc = device.createCommandEncoder();
const pass = enc.beginComputePass();
pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1);
pass.end();
enc.copyBufferToBuffer(bufC, 0, read, 0, M * N * 4);
device.queue.submit([enc.finish()]);
await read.mapAsync(GPUMapMode.READ);
const out = new Uint32Array(read.getMappedRange().slice(0));
read.unmap();
let bad = 0;
for (let i = 0; i < M * N; i++) if (out[i] !== ref[i]) bad++;
console.log(bad === 0 ? 'CORRECTNESS PASS' : `MISMATCH ${bad}/${M * N}; out[0..7]=${[...out.slice(0,8)]} ref=${[...ref.slice(0,8)]}`);
