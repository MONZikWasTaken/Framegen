# RIFE-Lite (RIFEm) reference — port ground truth

Extracted verbatim from `hzwer/ECCV2022-RIFE` (branch `main`). The Rust port in
`src/model.rs` must reproduce this numerically. Do not eyeball — match these ops.

## Variant

`model.RIFE.Model(arbitrary=True)` -> `self.flownet = IFNet_m()` (file `model/IFNet_m.py`).
Inference-time call: `Model.inference(img0, img1, scale=1, scale_list=[4,2,1], timestep=0.5)`
returns `merged[2]` (TTA off). Input images are RGB in **[0,1]**, NCHW.

`IFNet_m.forward(x, scale=[4,2,1], timestep=0.5)`:
- `x` is `cat(img0, img1)` = 6 channels at inference (no gt; training-only branches
  gated by `gt.shape[1] == 3`). `gt` slice = `x[:, 6:]` -> 0 channels -> teacher block skipped.
- `timestep` broadcast: `(x[:, :1].clone()*0 + 1) * timestep` -> tensor [B,1,H,W] filled with `timestep`.

## warp — `model/warplayer.py` (backward warp)

```
g = normalized_meshgrid(H,W) + flow_normalized      # flow scaled to [-1,1] pixel coords
flow[:,0] /= (W-1)/2     flow[:,1] /= (H-1)/2
grid_sample(input, grid=NHWC(g), mode='bilinear', padding_mode='border', align_corners=True)
```
meshgrid: x = linspace(-1,1,W) (per col), y = linspace(-1,1,H) (per row); cat along C, expand to B.
align_corners=True on grid_sample; flow normalization uses (W-1)/2 — consistent with align_corners=True.

## IFBlock

```
conv(in,out,k=3,s,p=1,d=1) = Conv2d(in,out,k,s,p,d, bias=True) ; PReLU(out)
deconv(in,out,k=4,s=2,p=1) = ConvTranspose2d(in,out,k=4,s=2,p=1) ; PReLU(out)

IFBlock(in_planes, c):
  conv0    = [ conv(in_planes, c//2, k=3,s=2,p=1), conv(c//2, c, k=3,s=2,p=1) ]   # downsample /4
  convblock= [ conv(c,c) x8 ]
  lastconv = ConvTranspose2d(c, 5, k=4, s=2, p=1)                               # upsample /2 -> /2

IFBlock.forward(x, flow, scale):
  if scale != 1:  x = interpolate(x, 1/scale, bilinear, align_corners=False)
  if flow is not None:
      flow = interpolate(flow, 1/scale, bilinear, align_corners=False) * (1/scale)
      x = cat(x, flow, dim=1)
  x = conv0(x)
  x = convblock(x) + x            # residual: add input of convblock (8 convs) to its output
  tmp = lastconv(x)
  tmp = interpolate(tmp, scale*2, bilinear, align_corners=False)
  flow = tmp[:, :4] * (scale*2)
  mask = tmp[:, 4:5]
  return flow, mask
```
Note: residual `convblock(x)+x` adds the **input** of convblock (== output of conv0), not raw x.

## IFNet_m forward (inference, gt=None)

```
block0 = IFBlock(6+1,   c=240)
block1 = IFBlock(13+4+1, c=150)
block2 = IFBlock(13+4+1, c=90)
block_tea = IFBlock(16+4+1, c=90)   # UNUSED at inference (needs gt)
contextnet, unet  (see refine.py)

flow = None
warped_img0 = img0; warped_img1 = img1
stu = [block0, block1, block2]
for i in 0..3:
    if flow is not None:
        inp = cat(img0, img1, timestep, warped_img0, warped_img1, mask, dim=1)   # =13+1=14 ch
        flow_d, mask_d = stu[i](inp, flow, scale=scale[i])                        # +flow(4)=18 inside
        flow = flow + flow_d
        mask = mask + mask_d
    else:                                          # i==0
        flow, mask = stu[i](cat(img0,img1,timestep,dim=1)=7ch, None, scale=scale[i])
    mask_list[i] = sigmoid(mask)
    flow_list[i]  = flow
    warped_img0 = warp(img0, flow[:, :2])
    warped_img1 = warp(img1, flow[:, 2:4])
    merged[i] = (warped_img0, warped_img1)          # tuple, resolved below

for i in 0..3:
    merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])

# refinement (inference-only branch):
c0 = contextnet(img0, flow[:, :2])
c1 = contextnet(img1, flow[:, 2:4])
tmp = unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)   # mask = RAW mask here
res = tmp[:, :3] * 2 - 1
merged[2] = clamp(merged[2] + res, 0, 1)

return merged[2]
```
Note: unet receives the **raw** mask tensor (pre-sigmoid), and `flow` (4-ch) is the final
refined student flow. `mask_list[2]` uses sigmoid; unet input uses raw mask.

## refine.py

```
c = 16
Conv2(in,out,stride=2): conv1 = conv(in,out,3,stride,1); conv2 = conv(out,out,3,1,1)
Contextnet:
  conv1 = Conv2(3, 16); conv2 = Conv2(16,32); conv3 = Conv2(32,64); conv4 = Conv2(64,128)
  forward(x, flow):                   # flow has 2 channels
    for each level: x = conv_i(x); flow = interpolate(flow, 0.5, bilinear, align_corners=False)*0.5
                    f_i = warp(x, flow)
    return [f1, f2, f3, f4]

Unet:
  down0 = Conv2(17, 32); down1 = Conv2(64,64); down2 = Conv2(128,128); down3 = Conv2(256,256)
  up0 = deconv(512, 128); up1 = deconv(256, 64); up2 = deconv(128, 32); up3 = deconv(64, 16)
  conv = Conv2d(16, 3, 3, 1, 1)
  forward(img0,img1,w0,w1,mask,flow,c0,c1):
    s0 = down0(cat(img0,img1,w0,w1,mask,flow))        # 3+3+3+3+1+4 = 17
    s1 = down1(cat(s0, c0[0], c1[0]))                  # 32+16+16 = 64
    s2 = down2(cat(s1, c0[1], c1[1]))                  # 64+32+32 = 128
    s3 = down3(cat(s2, c0[2], c1[2]))                  # 128+64+64 = 256
    x  = up0(cat(s3, c0[3], c1[3]))                    # 256+128+128 = 512
    x  = up1(cat(x, s2))                                # 128+128 = 256
    x  = up2(cat(x, s1))                                # 64+64 = 128
    x  = up3(cat(x, s0))                                # 32+32 = 64
    x  = conv(x)                                        # -> 3
    return sigmoid(x)
```

## Channel counts (sanity)

| tensor | channels |
|---|---|
| img0 / img1 | 3 |
| timestep | 1 |
| flow (per direction 2) | 4 |
| mask (raw) | 1 |
| block0 input | 7 (=6+1) |
| block1/2 input (+flow inside) | 14+4=18 (=13+4+1) |
| unet input | 17 |

## Preprocessing

Images arrive as [0,1] float NCHW, RGB. No normalization beyond that. Output same range,
clamped. ffmpeg harness must decode to RGB8 -> /255 -> f32; reverse on encode.
