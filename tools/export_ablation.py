"""Export training-free ablation variants of RIFE-Lite (IFNet_m) to ONNX.

Phase 5 step 1: before distilling, measure which parts of the net we can cut for free.
Variants (all fixed-shape, timestep=0.5):
  noref       - skip contextnet+unet refinement (output = warped blend)
  s842        - run the 3 IFBlocks at scales [8,4,2] instead of [4,2,1] (~4x cheaper blocks)
  s842_noref  - both
  2blk        - drop block2 (the most expensive IFBlock), scales [4,2], keep refinement
  2blk_noref  - drop block2 and refinement

Usage: python export_ablation.py [H W] [variant ...]   (default 720 1280, all variants)
"""
import os
import sys

import torch

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
from model.RIFE import Model
from model.warplayer import warp

args = sys.argv[1:]
H, W = (int(args[0]), int(args[1])) if len(args) >= 2 else (720, 1280)
VARIANTS = args[2:] or ["noref", "s842", "s842_noref", "2blk", "2blk_noref"]

# FRAMECAST_WEIGHTS overrides the checkpoint dir (must contain flownet.pkl) —
# used to export trained students from tools/train_student.py.
WEIGHTS = os.environ.get("FRAMECAST_WEIGHTS") or os.path.join(
    os.environ["TEMP"], "opencode", "rife_m", "RIFE_m_train_log")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

torch.set_grad_enabled(False)
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})

model = Model(arbitrary=True)
model.load_model(WEIGHTS, -1)
model.eval()
model.device()
net = model.flownet


class Ablated(torch.nn.Module):
    """Inference-only IFNet_m forward with configurable scales / block count / refinement."""

    def __init__(self, net, scales, nblocks, refine):
        super().__init__()
        self.net = net
        self.scales = scales
        self.nblocks = nblocks
        self.refine = refine

    def forward(self, img0, img1):
        timestep = (img0[:, :1] * 0 + 1) * 0.5
        stu = [self.net.block0, self.net.block1, self.net.block2]
        flow = None
        mask = None
        warped_img0 = img0
        warped_img1 = img1
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
            res = tmp[:, :3] * 2 - 1
            merged = torch.clamp(merged + res, 0, 1)
        return merged


CONFIGS = {
    "noref":      dict(scales=[4, 2, 1], nblocks=3, refine=False),
    "s842":       dict(scales=[8, 4, 2], nblocks=3, refine=True),
    "s842_noref": dict(scales=[8, 4, 2], nblocks=3, refine=False),
    "2blk":       dict(scales=[4, 2],    nblocks=2, refine=True),
    "2blk_noref": dict(scales=[4, 2],    nblocks=2, refine=False),
}

ph = ((H - 1) // 32 + 1) * 32
pw = ((W - 1) // 32 + 1) * 32
dummy0 = torch.randn(1, 3, ph, pw)
dummy1 = torch.randn(1, 3, ph, pw)

from model import warplayer

for name in VARIANTS:
    warplayer.backwarp_tenGrid.clear()  # cached grids leak FakeTensors between dynamo exports
    cfg = CONFIGS[name]
    wrapper = Ablated(net, **cfg)
    suffix = os.environ.get("FRAMECAST_SUFFIX", "")  # e.g. "_student" for trained checkpoints
    out_path = os.path.join(OUT_DIR, f"rife_lite_{H}p_{name}{suffix}.onnx")
    torch.onnx.export(
        wrapper, (dummy0, dummy1), out_path,
        export_params=True, opset_version=20, do_constant_folding=True,
        input_names=["img0", "img1"], output_names=["mid"],
        dynamic_axes=None, verbose=False, dynamo=True,
        optimize=False,  # onnxscript RewritePass crashes on the s842 graph (.numpy() on tensor subclass)
    )
    import onnx as _onnx
    m = _onnx.load(out_path, load_external_data=True)
    _onnx.save_model(m, out_path, save_as_external_data=False)
    print(f"{name}: {out_path} ({os.path.getsize(out_path) // 1024 // 1024} MB) shape=({ph},{pw})")
