# Experiment: batch two mids per dispatch (GPU occupancy at low rungs)

Problem (measured): the /16-grid head convs at 480p are 53x30 points = a few
dozen workgroups per z-block - a 34-SM card is underfed, so 360p is only ~2x
faster than 720p instead of ~4x.

Plan: give every /16- and /8-grid head stage a batch dimension of 2:
- film / cb6 / cb7 / deconv (+ the /4 refine chain) index a second activation
  plane at offset = batch * planeSize; dispatch z doubles.
- runT gains runT2(tA, tB, outA, outB); flowout stays per-t (full-res grid is
  not starved). The JIT scheduler submits pairs when two mids fit the lead
  window, single mids otherwise.
- Success bar: >=15% lower per-mid cost at 480p and below, parity max|d|<=2 vs
  main, no regression at 720p+. Otherwise the branch dies.

Status: design only - implementation needs an idle GPU for bench-driven work.

## Result (2026-07-04, 4060 Ti, idle GPU) - EXPERIMENT KILLED

Implemented: full B-set of head buffers + runT2 interleaving A/B chains
stage-by-stage in one compute pass. Parity: max|d|=0 (bit-identical).

Measured amortized ms/mid, single runT vs paired runT2:
- 640x352: 0.83 -> 0.82 (2%)
- 848x480: 1.29 -> 1.24 (4%)
- 1280x720: 2.10 -> 2.07 (2%)

Success bar was >=15% at 480p and below. 2-4% means the driver does NOT
meaningfully overlap the interleaved independent chains (D3D12 UAV barriers
between dispatches appear to serialize the pass anyway), and back-to-back
queue pressure already hides most launch gaps. Branch stays as an archive;
do not merge. The occupancy theory for low rungs is hereby measured dead -
remaining speed levers are subgroups (exp/subgroups) and subgroup-matrix.
