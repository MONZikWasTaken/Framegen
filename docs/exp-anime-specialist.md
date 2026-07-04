# Experiment: anime specialist (ATD-12K finetune) — killed

**Date:** 2026-07-04 · **Hardware:** rented RTX 4090 (vast.ai) · **Cost:** ~$0.50

## Hypothesis

A finetune of the v0.6.0 tfact2 checkpoint on ATD-12K (12k curated animation
triplets — a genuinely different data distribution, unlike the failed film
specialist) beats the universal model on anime by ≥ +0.3 dB, the agreed bar
for shipping a content specialist plus a selector.

## Setup

- `tools/train_anime_spec.py`: alternating steps — GT+teacher loss at t=0.5
  (ATD triplets), teacher-only distill at off-center t (keeps timestep
  conditioning alive; ATD has no GT off t=0.5).
- 20k steps planned, batch 32, lr 3e-5 cosine, RIFE_m teacher.
- Eval: 100 triplets from ATD test_2k_540p; BBB/jellyfish as the canary.

## Result: killed at step 12600 of 20000

| eval | ATD (init 28.71) | BBB canary (init 39.94) |
|------|------------------|-------------------------|
| 1000 | 28.75 | 39.85 |
| 4000 | 28.80 | 39.74 |
| 6000 | 28.83 | 39.64 |
| 8000 | **28.84** (best) | 39.71 |
| 10000 | 28.82 | 39.63 |
| 12000 | 28.84 | 39.67 |

+0.13 dB, six consecutive evals flat at 28.82–28.84 from the 30% mark.
Extrapolated finish ≈ +0.15–0.2 — under the +0.3 bar. Canary cost −0.2 to
−0.3 dB on BBB.

## Why it plateaued (best current understanding)

The universal v0.6.0 model already generalizes to anime reasonably well; the
gap ATD closes is small. The hard anime failure modes (large flat fills,
sharp cuts, motion on twos with huge displacement) are capacity/architecture
limits of the small student, not a data-distribution gap — more anime data
cannot buy what the trunk cannot represent.

## Consequences

- No anime specialist ships; the content selector stays shelved (nothing to
  select between).
- Checkpoint + log on the shelf: `E:\data\framecast\ckpt_anime_atd\`.
- The anime path forward is the v7 track (stronger teacher, larger student),
  not more finetuning at current capacity.
- ATD-12K itself is downloaded and staged (Kaggle mirror; GDrive quota is
  permanently hostile) — reusable for v7 training/eval.
