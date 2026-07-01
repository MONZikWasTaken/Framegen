"""Export RIFE-Lite (IFNet_m) to ONNX for TensorRT conversion.

Output: rife_lite.onnx with:
  inputs:  img0 [1,3,H,W] float32, img1 [1,3,H,W] float32
  output:  mid  [1,3,H,W] float32 (interpolated frame at timestep 0.5)

Fixed timestep=0.5 for 2x interpolation (most common case).
Dynamic batch=1 (real-time inference).
"""
import sys
import os
import torch

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
from model.RIFE import Model

WEIGHTS = r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_m\RIFE_m_train_log"
H, W = int(sys.argv[1]) if len(sys.argv) > 1 else 720, int(sys.argv[2]) if len(sys.argv) > 2 else 1280
OUT_ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", f"rife_lite_{H}x{W}.onnx")

torch.set_grad_enabled(False)
device = torch.device("cpu")

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})

model = Model(arbitrary=True)
model.load_model(WEIGHTS, -1)
model.eval()
model.device()


class RifeWrapper(torch.nn.Module):
    """Thin wrapper: takes img0, img1 separately (not concatenated), returns mid frame."""
    def __init__(self, rife_model):
        super().__init__()
        self.rife = rife_model.flownet

    def forward(self, img0, img1):
        x = torch.cat((img0, img1), 1)
        scale_list = [4, 2, 1]
        timestep = 0.5
        flow_list, mask_list, merged, flow_teacher, merged_teacher, loss_distill = \
            self.rife(x, scale=scale_list, timestep=timestep)
        return merged[2]


wrapper = RifeWrapper(model)

# Pad to multiple of 32
ph = ((H - 1) // 32 + 1) * 32
pw = ((W - 1) // 32 + 1) * 32
dummy0 = torch.randn(1, 3, ph, pw)
dummy1 = torch.randn(1, 3, ph, pw)

# Export directly (no jit.trace — warplayer has dynamic grid_sample)
os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
torch.onnx.export(
    wrapper,
    (dummy0, dummy1),
    OUT_ONNX,
    export_params=True,
    opset_version=20,
    do_constant_folding=True,
    input_names=["img0", "img1"],
    output_names=["mid"],
    dynamic_axes=None,
    verbose=False,
    dynamo=True,  # use TorchDynamo exporter (handles dynamic code)
)

# Inline external data into the ONNX file (for TensorRT compatibility)
import onnx as _onnx
m = _onnx.load(OUT_ONNX, load_external_data=True)
_onnx.save_model(m, OUT_ONNX, save_as_external_data=False)
print(f"exported ONNX: {OUT_ONNX} ({os.path.getsize(OUT_ONNX) // 1024 // 1024} MB)")
print(f"shape: ({ph}, {pw}) — {'padded' if ph != H or pw != W else 'exact'}")
