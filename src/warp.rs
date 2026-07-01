#[cfg_attr(not(feature = "cuda"), allow(unused_imports, dead_code))]
use candle_core::{
    backend::BackendDevice, CpuStorage, CustomOp2, CudaStorage, DType, Layout, Result, Shape,
    Tensor,
};

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const WARP_CUDA: &str = r#"
extern "C" __global__ void backward_warp_bilinear(
    const float* __restrict__ input,
    const float* __restrict__ flow,
    float* __restrict__ output,
    int B, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    int rem = idx;
    int w = rem % W;  rem /= W;
    int h = rem % H;  rem /= H;
    int c = rem % C;  rem /= C;
    int b = rem;

    int flow_idx = (b * 2) * H * W + h * W + w;
    float fx = flow[flow_idx];
    float fy = flow[flow_idx + H * W];

    float src_x = fminf(fmaxf((float)w + fx, 0.0f), (float)(W - 1));
    float src_y = fminf(fmaxf((float)h + fy, 0.0f), (float)(H - 1));

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);

    float wx = src_x - (float)x0;
    float wy = src_y - (float)y0;

    int base = (b * C + c) * H * W;
    float v00 = input[base + y0 * W + x0];
    float v01 = input[base + y0 * W + x1];
    float v10 = input[base + y1 * W + x0];
    float v11 = input[base + y1 * W + x1];

    output[idx] = (1.0f - wx) * (1.0f - wy) * v00
                + wx * (1.0f - wy) * v01
                + (1.0f - wx) * wy * v10
                + wx * wy * v11;
}
"#;

#[derive(Debug, Clone)]
struct BackwardWarpOp;

impl CustomOp2 for BackwardWarpOp {
    fn name(&self) -> &'static str {
        "backward_warp_bilinear"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let input = match s1 {
            CpuStorage::F32(v) => v,
            _ => candle_core::bail!("backward_warp: input must be f32"),
        };
        let flow = match s2 {
            CpuStorage::F32(v) => v,
            _ => candle_core::bail!("backward_warp: flow must be f32"),
        };
        let (b, c, h, w) = l1.shape().dims4()?;
        let (bf, cf, hf, wf) = l2.shape().dims4()?;
        if cf != 2 || hf != h || wf != w || bf != b {
            candle_core::bail!(
                "backward_warp: flow shape {:?} mismatch with input {:?}",
                l2.shape(),
                l1.shape()
            );
        }
        let in_offset = l1.start_offset();
        let flow_offset = l2.start_offset();
        let mut out = vec![0f32; b * c * h * w];
        for bi in 0..b {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let fi = flow_offset + (bi * 2) * h * w + hi * w + wi;
                        let fx = flow[fi];
                        let fy = flow[fi + h * w];
                        let sx = (wi as f32 + fx).clamp(0.0, (w - 1) as f32);
                        let sy = (hi as f32 + fy).clamp(0.0, (h - 1) as f32);
                        let x0 = sx as usize;
                        let y0 = sy as usize;
                        let x1 = (x0 + 1).min(w - 1);
                        let y1 = (y0 + 1).min(h - 1);
                        let wx = sx - x0 as f32;
                        let wy = sy - y0 as f32;
                        let base = in_offset + (bi * c + ci) * h * w;
                        let v00 = input[base + y0 * w + x0];
                        let v01 = input[base + y0 * w + x1];
                        let v10 = input[base + y1 * w + x0];
                        let v11 = input[base + y1 * w + x1];
                        let val = (1.0 - wx) * (1.0 - wy) * v00
                            + wx * (1.0 - wy) * v01
                            + (1.0 - wx) * wy * v10
                            + wx * wy * v11;
                        let oi = (bi * c + ci) * h * w + hi * w + wi;
                        out[oi] = val;
                    }
                }
            }
        }
        Ok((CpuStorage::F32(out), (b, c, h, w).into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::{cudarc, WrapErr};
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let (b, c, h, w) = l1.shape().dims4()?;
        let dev = s1.device.clone();
        let input = s1.as_cuda_slice::<f32>()?;
        let flow = s2.as_cuda_slice::<f32>()?;

        let out_shape: Shape = (b, c, h, w).into();
        let out = unsafe { dev.alloc_uninit(&out_shape, DType::F32)? };
        let out_slice = out.as_cuda_slice::<f32>()?;

        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(
            WARP_CUDA,
            cudarc::nvrtc::CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            },
        )
        .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;
        let ptx_src = ptx.to_src();

        let func = dev.get_or_load_custom_func(
            "backward_warp_bilinear",
            "backward_warp_bilinear",
            &ptx_src,
        )?;

        let elem = b * c * h * w;
        let cfg = LaunchConfig {
            grid_dim: (((elem + 255) / 255) as u32, 1, 1),
            block_dim: (255, 1, 1),
            shared_mem_bytes: 0,
        };

        let in_off = l1.start_offset();
        let flow_off = l2.start_offset();
        let stream = dev.cuda_stream();
        let in_view = input.slice(in_off..in_off + elem);
        let flow_view = flow.slice(flow_off..flow_off + b * 2 * h * w);
        let bi = b as i32;
        let ci = c as i32;
        let hi = h as i32;
        let wi = w as i32;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&in_view);
        builder.arg(&flow_view);
        builder.arg(out_slice);
        builder.arg(&bi);
        builder.arg(&ci);
        builder.arg(&hi);
        builder.arg(&wi);
        unsafe { builder.launch(cfg) }.w()?;
        Ok((out, out_shape))
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        candle_core::bail!("backward_warp: cuda not available (build with --features cuda)")
    }
}

pub fn warp(input: &Tensor, flow: &Tensor) -> Result<Tensor> {
    let dt = input.dtype();
    // warp kernel is f32-only; cast to f32, run, cast back.
    if dt == DType::F32 {
        input.apply_op2_no_bwd(flow, &BackwardWarpOp)
    } else {
        let in_f32 = input.to_dtype(DType::F32)?;
        let flow_f32 = flow.to_dtype(DType::F32)?;
        let out_f32 = in_f32.apply_op2_no_bwd(&flow_f32, &BackwardWarpOp)?;
        out_f32.to_dtype(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn warp_identity() {
        let dev = Device::Cpu;
        let img = Tensor::arange(0f32, 12f32, &dev)
            .unwrap()
            .reshape((1, 1, 3, 4))
            .unwrap();
        let flow = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev).unwrap();
        let out = warp(&img, &flow).unwrap();
        let a = img.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b);
    }
}
