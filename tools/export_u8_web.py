"""Export RIFE variants for the BROWSER with canvas-native u8 RGBA I/O.

Inputs:  img0, img1  [1,H,W,4] uint8 RGBA (ImageData.data as-is, alpha ignored)
Output:  mid         [1,H,W,4] uint8 RGBA (alpha=255) — feed straight to putImageData

The graph does everything prepost did in JS: slice RGB -> cast/255 -> HWC->CHW ->
RGB->BGR -> pad to /32 -> net (timestep baked at 0.5) -> crop -> BGR->RGB ->
CHW->HWC -> *255 -> truncate -> append alpha. Zero per-pixel JS anywhere.

Env: FRAMECAST_WEIGHTS (dir with flownet.pkl), FRAMECAST_SUFFIX (output name suffix).
Usage: python export_u8_web.py <H> <W> <variant>
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
from model.RIFE import Model
from model.warplayer import warp

H, W = int(sys.argv[1]), int(sys.argv[2])
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "full"

WEIGHTS = os.environ.get("FRAMECAST_WEIGHTS") or os.path.join(
    os.environ["TEMP"], "opencode", "rife_m", "RIFE_m_train_log")
SUFFIX = os.environ.get("FRAMECAST_SUFFIX", "")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

CONFIGS = {
    "full":       dict(scales=[4, 2, 1], nblocks=3, refine=True),
    "fast":       dict(scales=[4, 2, 1], nblocks=3, refine=False),
    "fastest":    dict(scales=[4, 2],    nblocks=2, refine=False),
    "turbo":      dict(scales=[4],       nblocks=1, refine=False),
}

torch.set_grad_enabled(False)
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})
model = Model(arbitrary=True)
model.load_model(WEIGHTS, -1)
model.eval()
model.device()
net = model.flownet

PH = ((H - 1) // 32 + 1) * 32
PW = ((W - 1) // 32 + 1) * 32


class U8WebPipeline(torch.nn.Module):
    def __init__(self, net, scales, nblocks, refine):
        super().__init__()
        self.net = net
        self.scales = scales
        self.nblocks = nblocks
        self.refine = refine

    def core(self, img0, img1):
        timestep = (img0[:, :1] * 0 + 1) * 0.5
        stu = [self.net.block0, self.net.block1, self.net.block2][: self.nblocks]
        flow = None
        mask = None
        warped_img0, warped_img1 = img0, img1
        for i in range(self.nblocks):
            if flow is not None:
                flow_d, mask_d = stu[i](
                    torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1),
                    flow, scale=self.scales[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1, timestep), 1), None,
                                    scale=self.scales[i])
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        merged = warped_img0 * m + warped_img1 * (1 - m)
        if self.refine:
            c0 = self.net.contextnet(img0, flow[:, :2])
            c1 = self.net.contextnet(img1, flow[:, 2:4])
            tmp = self.net.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            merged = torch.clamp(merged + (tmp[:, :3] * 2 - 1), 0, 1)
        return merged

    def forward(self, img0_u8, img1_u8):  # [1,H,W,4] uint8 RGBA
        def prep(u8):
            x = u8[..., :3].to(torch.float32).div(255.0).permute(0, 3, 1, 2)  # RGB CHW
            x = x.flip(1)                                                     # -> BGR
            return F.pad(x, (0, PW - W, 0, PH - H))
        mid = self.core(prep(img0_u8), prep(img1_u8))            # [1,3,PH,PW] BGR 0..1
        mid = mid[:, :, :H, :W].flip(1)                          # crop, BGR -> RGB
        mid = mid.clamp(0, 1).mul(255.0).permute(0, 2, 3, 1)     # [1,H,W,3]
        alpha = torch.full_like(mid[..., :1], 255.0)
        return torch.cat([mid, alpha], dim=-1).to(torch.uint8)   # [1,H,W,4] RGBA


wrapper = U8WebPipeline(net, **CONFIGS[VARIANT])
d0 = torch.randint(0, 256, (1, H, W, 4), dtype=torch.uint8)
d1 = torch.randint(0, 256, (1, H, W, 4), dtype=torch.uint8)

out_path = os.path.join(OUT_DIR, f"rife_web_{H}x{W}_{VARIANT}{SUFFIX}.onnx")
torch.onnx.export(
    wrapper, (d0, d1), out_path,
    export_params=True, opset_version=20, do_constant_folding=True,
    input_names=["img0", "img1"], output_names=["mid"],
    dynamic_axes=None, verbose=False, dynamo=True, optimize=False,
)
import onnx as _onnx
m = _onnx.load(out_path, load_external_data=True)
_onnx.save_model(m, out_path, save_as_external_data=False)
print(f"{VARIANT}: {out_path} ({os.path.getsize(out_path) // 1024 // 1024} MB) io=[1,{H},{W},4]u8")
