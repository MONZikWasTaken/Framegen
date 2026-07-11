# subgroup-matrix (tensor cores) exploration

Probing WebGPU's experimental `chromium_experimental_subgroup_matrix` extension
- GPU tensor cores (NVIDIA cooperative matrix / Metal simdgroup matrix) - as a
path to make Framegen's f16 convs faster.

## Verdict (2026-07-08, RTX 4060 Ti): tensor cores give ~5x on the GEMM, and
## the path IS open - but not in a shipping browser yet.

Measured, f16xf16->f32 tensor GEMM at the trunk conv shape (M=192 out-channels,
K=1728 = 192in x 3x3, N=1600 pixels):

| kernel | time | vs ours |
|---|---|---|
| our tuned f16 subgroup conv (same shape) | 0.530 ms | 1x |
| **f16 tensor-core GEMM (naive, nacc=2)** | **0.108 ms** | **4.9x** |

Correctness: 0.00% max relative error vs a CPU f32 reference. And this is a
*naive* tensor kernel (no shared-memory reuse) hitting only ~10 TFLOPS of the
card's ~100+ f16 tensor TFLOPS - a tiled kernel would go faster still.

Caveat: this is the GEMM in isolation. A real conv also needs im2col (input
repacking) and the input/output bandwidth, which our 0.53 ms number already
includes. The realistic end-to-end win is smaller than 4.9x but the trunk is
GEMM-dominated, so it's real and large.

## Why it "didn't work" at first, and the toggle that unlocked it

f16 + subgroup-matrix looked mutually exclusive until we found the cause. Dawn
**intentionally disables f16 on NVIDIA/Vulkan** unless a dev toggle is set:

```cpp
// src/dawn/native/vulkan/PhysicalDeviceVk.cpp, ValidateFeatureSupportedWithTogglesImpl
case wgpu::FeatureName::ShaderF16:
  // TODO(crbug.com/42251215): Investigate f16 CTS test failures to enable on Nvidia.
  if (gpu_info::IsNvidia(mVendorId) && !toggles.IsEnabled(Toggle::VulkanEnableF16OnNvidia))
    return "Feature ShaderF16 is not yet supported on Nvidia GPUs";
```

Enabling `vulkan_enable_f16_on_nvidia` gives shader-f16 + subgroup-matrix
together, and the adapter then exposes f16 configs: `f16 16x16x16 -> f16/f32`.
The predicate that *enables* ShaderF16 in the same file all reads true on this
card (verified with an instrumented build: hasExt/shaderFloat16/shaderInt16/
storageBuffer16 all = 1). So this is a deliberate NVIDIA block behind a CTS
TODO, not a bug.

## Practical status for Framegen

- **Not usable in users' browsers today.** subgroup-matrix is experimental
  (needs allow_unsafe_apis) and f16-on-NVIDIA/Vulkan is dev-toggle-gated. Real
  users get neither. When Google ships both, we're ready.
- Our f16 correctness result (0.00% error on the GEMM) is a small data point
  *for* lifting the NVIDIA f16 block - could go on crbug.com/42251215.

## Reproduce

Needs a local Dawn `dawn.node` (main) built Vulkan-only; see
`memory/tensor-cores-webgpu.md` for the full build recipe. Then:

```
node gemm_f16.mjs <path-to-dawn.node> 2     # f16 tensor GEMM + correctness
node gemm_int8.mjs                          # INT8 path (works on prebuilt too)
```
Run flags baked in: `enable-dawn-features=allow_unsafe_apis,vulkan_enable_f16_on_nvidia`,
`backend=vulkan`, `adapter=NVIDIA` (else it grabs SwiftShader). Copy the system
vulkan-1.dll next to dawn.node on Windows.

## Findings for Dawn (contributor angles)

1. Prebuilt dawn.node on Windows: vulkan-1.dll not found next to the binary
   (loader looks beside the .node, not on PATH). Small, real - CL candidate.
2. Teardown segfault after creating a timestamp QuerySet on Vulkan (prebuilt
   March build). Needs confirming on main.
3. f16-on-NVIDIA block (crbug.com/42251215): our correct f16 GEMM is evidence,
   but lifting it means passing the CTS - not a drive-by CL.
