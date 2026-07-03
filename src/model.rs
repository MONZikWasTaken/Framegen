use candle_core::{DType, Result, Tensor};
use candle_nn::{
    conv2d, conv_transpose2d, ops::sigmoid, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig,
    Module, PReLU, VarBuilder,
};

use crate::warp::warp;

// conv(in,out,k,s,p,d) = Conv2d(bias) ; PReLU(out)  - port of model/IFNet_m.py::conv
struct ConvP {
    conv: Conv2d,
    act: PReLU,
}

impl ConvP {
    fn new(in_p: usize, out: usize, k: usize, s: usize, p: usize, d: usize, vb: VarBuilder) -> Result<Self> {
        let conv = conv2d(in_p, out, k, Conv2dConfig { padding: p, stride: s, dilation: d, ..Default::default() }, vb.pp("0"))?;
        let act = candle_nn::prelu(Some(out), vb.pp("1"))?;
        Ok(Self { conv, act })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.act.forward(&self.conv.forward(x)?)
    }
}

// deconv(in,out,k=4,s=2,p=1) = ConvTranspose2d(bias) ; PReLU(out)
struct DeconvP {
    conv: ConvTranspose2d,
    act: PReLU,
}

impl DeconvP {
    fn new(in_p: usize, out: usize, k: usize, s: usize, p: usize, vb: VarBuilder) -> Result<Self> {
        let conv = conv_transpose2d(in_p, out, k, ConvTranspose2dConfig { padding: p, stride: s, dilation: 1, output_padding: 0 }, vb.pp("0"))?;
        let act = candle_nn::prelu(Some(out), vb.pp("1"))?;
        Ok(Self { conv, act })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.act.forward(&self.conv.forward(x)?)
    }
}

// Conv2(in,out,stride=2) = conv1: ConvP(in,out,s) ; conv2: ConvP(out,out,1) - port of refine.py::Conv2
struct Conv2b {
    conv1: ConvP,
    conv2: ConvP,
}

impl Conv2b {
    fn new(in_p: usize, out: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = ConvP::new(in_p, out, 3, stride, 1, 1, vb.pp("conv1"))?;
        let conv2 = ConvP::new(out, out, 3, 1, 1, 1, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        self.conv2.forward(&x)
    }
}

// Port of IFNet_m.py::IFBlock
struct IFBlock {
    conv0: Vec<ConvP>,      // 2 layers, stride-2 each -> /4 downsample
    convblock: Vec<ConvP>,  // 8 residual convs
    lastconv: ConvTranspose2d,
}

impl IFBlock {
    fn new(in_planes: usize, c: usize, vb: VarBuilder) -> Result<Self> {
        let conv0 = vec![
            ConvP::new(in_planes, c / 2, 3, 2, 1, 1, vb.pp("conv0").pp("0"))?,
            ConvP::new(c / 2, c, 3, 2, 1, 1, vb.pp("conv0").pp("1"))?,
        ];
        let mut convblock = Vec::with_capacity(8);
        for i in 0..8 {
            convblock.push(ConvP::new(c, c, 3, 1, 1, 1, vb.pp("convblock").pp(i.to_string()))?);
        }
        let lastconv = conv_transpose2d(c, 5, 4, ConvTranspose2dConfig { padding: 1, stride: 2, dilation: 1, output_padding: 0 }, vb.pp("lastconv"))?;
        Ok(Self { conv0, convblock, lastconv })
    }

    fn forward(&self, x: &Tensor, flow: Option<&Tensor>, scale: f64) -> Result<(Tensor, Tensor)> {
        let inv = 1.0 / scale;
        let x = x.upsample_bilinear2d_with_scale(inv, inv, false)?;
        let x = match flow {
            Some(f) => {
                let fd = f
                    .upsample_bilinear2d_with_scale(inv, inv, false)?
                    .affine(inv, 0.0)?;
                Tensor::cat(&[&x, &fd], 1)?
            }
            None => x,
        };

        let mut x = self.conv0[0].forward(&x)?;
        x = self.conv0[1].forward(&x)?;
        let skip = x.clone();
        for layer in &self.convblock {
            x = layer.forward(&x)?;
        }
        x = x.broadcast_add(&skip)?; // convblock(x) + x  (x = conv0 output)

        let tmp = self
            .lastconv
            .forward(&x)?
            .upsample_bilinear2d_with_scale(scale * 2.0, scale * 2.0, false)?;
        let flow = tmp.narrow(1, 0, 4)?.affine(scale * 2.0, 0.0)?;
        let mask = tmp.narrow(1, 4, 1)?;
        Ok((flow, mask))
    }
}

// Port of refine.py::Contextnet (c = 16)
struct Contextnet {
    conv1: Conv2b,
    conv2: Conv2b,
    conv3: Conv2b,
    conv4: Conv2b,
}

impl Contextnet {
    const C: usize = 16;

    fn new(vb: VarBuilder) -> Result<Self> {
        let c = Self::C;
        Ok(Self {
            conv1: Conv2b::new(3, c, 2, vb.pp("conv1"))?,
            conv2: Conv2b::new(c, 2 * c, 2, vb.pp("conv2"))?,
            conv3: Conv2b::new(2 * c, 4 * c, 2, vb.pp("conv3"))?,
            conv4: Conv2b::new(4 * c, 8 * c, 2, vb.pp("conv4"))?,
        })
    }

    // flow has 2 channels at input resolution; returns [f1, f2, f3, f4] at /2,/4,/8,/16.
    fn forward(&self, x: &Tensor, flow: &Tensor) -> Result<[Tensor; 4]> {
        let x = self.conv1.forward(x)?;
        let flow = flow
            .upsample_bilinear2d_with_scale(0.5, 0.5, false)?
            .affine(0.5, 0.0)?;
        let f1 = warp(&x, &flow)?;
        let x = self.conv2.forward(&x)?;
        let flow = flow
            .upsample_bilinear2d_with_scale(0.5, 0.5, false)?
            .affine(0.5, 0.0)?;
        let f2 = warp(&x, &flow)?;
        let x = self.conv3.forward(&x)?;
        let flow = flow
            .upsample_bilinear2d_with_scale(0.5, 0.5, false)?
            .affine(0.5, 0.0)?;
        let f3 = warp(&x, &flow)?;
        let x = self.conv4.forward(&x)?;
        let flow = flow
            .upsample_bilinear2d_with_scale(0.5, 0.5, false)?
            .affine(0.5, 0.0)?;
        let f4 = warp(&x, &flow)?;
        Ok([f1, f2, f3, f4])
    }
}

// Port of refine.py::Unet (c = 16)
struct Unet {
    down0: Conv2b,
    down1: Conv2b,
    down2: Conv2b,
    down3: Conv2b,
    up0: DeconvP,
    up1: DeconvP,
    up2: DeconvP,
    up3: DeconvP,
    conv: Conv2d, // bare Conv2d (no PReLU)
}

impl Unet {
    const C: usize = 16;

    fn new(vb: VarBuilder) -> Result<Self> {
        let c = Self::C;
        let conv = conv2d(c, 3, 3, Conv2dConfig { padding: 1, stride: 1, ..Default::default() }, vb.pp("conv"))?;
        Ok(Self {
            down0: Conv2b::new(17, 2 * c, 2, vb.pp("down0"))?,
            down1: Conv2b::new(4 * c, 4 * c, 2, vb.pp("down1"))?,
            down2: Conv2b::new(8 * c, 8 * c, 2, vb.pp("down2"))?,
            down3: Conv2b::new(16 * c, 16 * c, 2, vb.pp("down3"))?,
            up0: DeconvP::new(32 * c, 8 * c, 4, 2, 1, vb.pp("up0"))?,
            up1: DeconvP::new(16 * c, 4 * c, 4, 2, 1, vb.pp("up1"))?,
            up2: DeconvP::new(8 * c, 2 * c, 4, 2, 1, vb.pp("up2"))?,
            up3: DeconvP::new(4 * c, c, 4, 2, 1, vb.pp("up3"))?,
            conv,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img0: &Tensor,
        img1: &Tensor,
        warped_img0: &Tensor,
        warped_img1: &Tensor,
        mask: &Tensor,
        flow: &Tensor,
        c0: &[Tensor; 4],
        c1: &[Tensor; 4],
    ) -> Result<Tensor> {
        let s0 = self
            .down0
            .forward(&Tensor::cat(&[img0, img1, warped_img0, warped_img1, mask, flow], 1)?)?;
        let s1 = self
            .down1
            .forward(&Tensor::cat(&[&s0, &c0[0], &c1[0]], 1)?)?;
        let s2 = self
            .down2
            .forward(&Tensor::cat(&[&s1, &c0[1], &c1[1]], 1)?)?;
        let s3 = self
            .down3
            .forward(&Tensor::cat(&[&s2, &c0[2], &c1[2]], 1)?)?;
        let x = self
            .up0
            .forward(&Tensor::cat(&[&s3, &c0[3], &c1[3]], 1)?)?;
        let x = self.up1.forward(&Tensor::cat(&[&x, &s2], 1)?)?;
        let x = self.up2.forward(&Tensor::cat(&[&x, &s1], 1)?)?;
        let x = self.up3.forward(&Tensor::cat(&[&x, &s0], 1)?)?;
        let x = self.conv.forward(&x)?;
        sigmoid(&x)
    }
}

// Port of IFNet_m.py::IFNet_m (inference path only; gt/teacher block loaded but unused)
pub struct IfNetM {
    block0: IFBlock,
    block1: IFBlock,
    block2: IFBlock,
    #[allow(dead_code)]
    block_tea: IFBlock,
    contextnet: Contextnet,
    unet: Unet,
    dtype: DType,
}

impl IfNetM {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        Ok(Self {
            block0: IFBlock::new(6 + 1, 240, vb.pp("block0"))?,
            block1: IFBlock::new(13 + 4 + 1, 150, vb.pp("block1"))?,
            block2: IFBlock::new(13 + 4 + 1, 90, vb.pp("block2"))?,
            block_tea: IFBlock::new(16 + 4 + 1, 90, vb.pp("block_tea"))?,
            contextnet: Contextnet::new(vb.pp("contextnet"))?,
            unet: Unet::new(vb.pp("unet"))?,
            dtype,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // imgs: [B,6,H,W] = cat(img0, img1), RGB in [0,1]. scale = [4,2,1].
    pub fn forward(&self, imgs: &Tensor, scale: &[f64; 3], timestep: f64) -> Result<Tensor> {
        let (b, _c, h, w) = imgs.dims4()?;
        let dev = imgs.device();

        let img0 = imgs.narrow(1, 0, 3)?;
        let img1 = imgs.narrow(1, 3, 3)?;
        let ts = Tensor::full(timestep as f32, (b, 1, h, w), dev)?.to_dtype(self.dtype)?;

        let mut flow: Option<Tensor> = None;
        let mut mask: Option<Tensor> = None;
        let mut warped_img0 = img0.clone();
        let mut warped_img1 = img1.clone();
        let mut merged = Vec::with_capacity(3);

        for (i, &sc) in scale.iter().enumerate() {
            let (new_flow, new_mask) = match &flow {
                Some(f) => {
                    let m = mask.as_ref().unwrap();
                    let inp = Tensor::cat(&[&img0, &img1, &ts, &warped_img0, &warped_img1, m], 1)?;
                    let (fd, md) = self.block(i, Some(f), &inp, sc)?;
                    (f.broadcast_add(&fd)?, m.broadcast_add(&md)?)
                }
                None => {
                    let inp = Tensor::cat(&[&img0, &img1, &ts], 1)?;
                    self.block(i, None, &inp, sc)?
                }
            };
            flow = Some(new_flow);
            mask = Some(new_mask);

            let mask_sig = sigmoid(mask.as_ref().unwrap())?;
            let f = flow.as_ref().unwrap();
            let new_w0 = warp(&img0, &f.narrow(1, 0, 2)?)?;
            let new_w1 = warp(&img1, &f.narrow(1, 2, 2)?)?;
            warped_img0 = new_w0.clone();
            warped_img1 = new_w1.clone();

            let inv = Tensor::ones_like(&mask_sig)?.broadcast_sub(&mask_sig)?;
            let merged_i = new_w0
                .broadcast_mul(&mask_sig)?
                .broadcast_add(&new_w1.broadcast_mul(&inv)?)?;
            merged.push(merged_i);
        }

        let flow = flow.unwrap();
        let mask = mask.unwrap();
        let c0 = self.contextnet.forward(&img0, &flow.narrow(1, 0, 2)?)?;
        let c1 = self.contextnet.forward(&img1, &flow.narrow(1, 2, 2)?)?;
        let tmp = self.unet.forward(
            &img0, &img1, &warped_img0, &warped_img1, &mask, &flow, &c0, &c1,
        )?;
        let res = tmp.narrow(1, 0, 3)?.affine(2.0, -1.0)?;
        let out = merged[2].clone().broadcast_add(&res)?.clamp(0.0, 1.0)?;
        Ok(out)
    }

    fn block(&self, i: usize, flow: Option<&Tensor>, inp: &Tensor, scale: f64) -> Result<(Tensor, Tensor)> {
        match i {
            0 => self.block0.forward(inp, flow, scale),
            1 => self.block1.forward(inp, flow, scale),
            2 => self.block2.forward(inp, flow, scale),
            _ => candle_core::bail!("invalid block index {i}"),
        }
    }
}
