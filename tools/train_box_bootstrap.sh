#!/usr/bin/env bash
# Framecast training-box bootstrap (Linux box, PyTorch, 2x GPU).
# Run after uploading code/ckpts to /workspace/framecast.
# Downloads datasets server-side (datacenter pipe), extracts frames.
set -euo pipefail
cd /workspace/framecast

echo "== apt/pip =="
apt-get update -qq && apt-get install -y -qq ffmpeg aria2 > /dev/null
pip install -q opencv-python-headless numpy gdown safetensors

echo "== GPU sanity bench (weed out throttled miners) =="
python - <<'PY'
import torch, time
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    x = torch.randn(4096, 4096, device='cuda')
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): x = (x @ x).clamp(-1, 1)
    torch.cuda.synchronize()
    print(f"GPU{i} {torch.cuda.get_device_name(i)}: {50*2*4096**3/(time.time()-t0)/1e12:.1f} TFLOPS fp32")
PY

echo "== movies (public mirrors) =="
mkdir -p movies frames ckpt
dl() { [ -f "movies/$2" ] || aria2c -q -x8 -o "movies/$2" "$1"; echo "movies/$2: $(du -m movies/$2 | cut -f1) MB"; }
dl https://media.xiph.org/tearsofsteel/tears_of_steel_720p.mov tears_of_steel_720p.mov
dl https://download.blender.org/ED/ED_1024.avi elephants_dream_1024.avi
dl https://media.xiph.org/sintel/sintel-720-surround.mp4 sintel_720p.mp4
for n in crowd_run_1080p50 park_joy_1080p50 in_to_tree_1080p50 old_town_cross_1080p50 ducks_take_off_1080p50; do
  dl "https://media.xiph.org/video/derf/y4m/$n.y4m" "real_$n.y4m"
done

echo "== ATD-12K (anime) - may fail on GDrive quota, non-fatal =="
gdown 1XBDuiEgdd6c0S4OXLF4QvgSn_XNPwc-g -O movies/atd12k.zip || echo "ATD blocked, continuing without"

echo "== frames =="
export PYTHONPATH=/workspace/framecast/rife_ref
python tools/extract_frames.py movies/sintel_720p.mp4 movies/tears_of_steel_720p.mov \
  movies/elephants_dream_1024.avi movies/real_*.y4m --out=/workspace/framecast

echo "== done. queues: =="
echo "GPU0 (big distill):  PYTHONPATH=/workspace/framecast/rife_ref CUDA_VISIBLE_DEVICES=0 python tools/train_tfact2.py --data /workspace/framecast/frames --out ckpt/big --teacher ckpt/flownet.pkl --slim-ckpt ckpt/student_last.pkl --tfact-ckpt ckpt/tfact_best.pt --steps 100000 --batch 32 --crop 320 --workers 12"
echo "GPU1 (specialists):  CUDA_VISIBLE_DEVICES=1 ... (film finetune etc.)"
