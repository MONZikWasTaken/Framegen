# Server-side: run EMA-VFI ours_t over the whole frames dataset and save its
# outputs as v7 distillation targets, mirroring the triplets.txt sampling that
# the student trainers use:
#   <clip>/<i:06d>_s2.jpg              t=0.5 of (i-1, i+1)
#   <clip>/<i:06d>_s4t{1,2,3}.jpg      t=k/4 of (i-2, i+2)   [when in range]
# One multi_inference call per pair shares the feature pyramid across t's.
# JPEG q97: recompression noise is far below the teacher-student gap.
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append('/workspace/EMA-VFI')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

DATA = '/workspace/frames_v060/frames'
OUT = '/workspace/tea_v7'

cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])
model = Model(-1)
model.load_model()
model.eval()
model.device()


def tens(img):
    return (torch.tensor(img.transpose(2, 0, 1).copy()).cuda() / 255.).unsqueeze(0)


@torch.no_grad()
def infer(a, b, ts):
    I0, I2 = tens(a), tens(b)
    padder = InputPadder(I0.shape, divisor=32)
    I0p, I2p = padder.pad(I0, I2)
    preds = model.multi_inference(I0p, I2p, TTA=True, time_list=ts, fast_TTA=True)
    return [(padder.unpad(p).clamp(0, 1).detach().cpu().numpy()
             .transpose(1, 2, 0) * 255.0).astype(np.uint8) for p in preds]


total_done = 0
t00 = time.time()
for stem in sorted(os.listdir(DATA)):
    d = os.path.join(DATA, stem)
    idx_file = os.path.join(d, 'triplets.txt')
    if not os.path.isfile(idx_file):
        continue
    od = os.path.join(OUT, stem)
    os.makedirs(od, exist_ok=True)
    idxs = [int(l) for l in open(idx_file) if l.strip()]
    mx = max(idxs) + 1  # frames 0..mx+1 exist
    rd = lambda k: cv2.imread(os.path.join(d, f'{k:06d}.jpg'))
    for n, i in enumerate(idxs):
        # resume: skip fully-done triplets
        s2 = os.path.join(od, f'{i:06d}_s2.jpg')
        s4ok = (i - 2 < 0 or i + 2 > mx + 1) or all(
            os.path.isfile(os.path.join(od, f'{i:06d}_s4t{k}.jpg')) for k in (1, 2, 3))
        if os.path.isfile(s2) and s4ok:
            continue
        a, b = rd(i - 1), rd(i + 1)
        if a is None or b is None:
            continue
        out = infer(a, b, [0.5])[0]
        cv2.imwrite(s2, out, [cv2.IMWRITE_JPEG_QUALITY, 97])
        if i - 2 >= 0 and i + 2 <= mx + 1:
            a4, b4 = rd(i - 2), rd(i + 2)
            if a4 is not None and b4 is not None:
                outs = infer(a4, b4, [0.25, 0.5, 0.75])
                for k, o in enumerate(outs, 1):
                    cv2.imwrite(os.path.join(od, f'{i:06d}_s4t{k}.jpg'), o,
                                [cv2.IMWRITE_JPEG_QUALITY, 97])
        total_done += 1
        if total_done % 100 == 0:
            el = time.time() - t00
            print(f'{stem} {n}/{len(idxs)} | total {total_done} trips '
                  f'| {el / total_done:.2f} s/trip', flush=True)

print(f'ALL_DONE {total_done} triplets in {(time.time() - t00) / 3600:.2f} h', flush=True)
