# Experiment: WGSL subgroups in the conv inner loops

Chrome ships the 'subgroups' WebGPU feature (not subgroup-matrix). Hypothesis:
subgroupBroadcast/subgroupShuffle can replace some shared-memory traffic and
barriers in wgslConvRB's weight/tile staging, worth 0-20%.

Plan: variant kernel behind a feature check (adapter.features.has('subgroups')),
bench against the autotuned baseline on the real conv shapes, parity-check
bit-identity is NOT expected (reduction order may change) - require max|d|<=2
on the rt_test harness instead.

Success bar: >=10% on at least one real rung on the 4060 Ti; otherwise die.

Status: design only.

## Result (2026-07-04, 4060 Ti, idle GPU) - SHIPPED VIA THE AUTOTUNER

Variant implemented: weights read straight from global via
subgroupBroadcastFirst (uniform index per wave), no shared staging for
weights, barrier covers the input tile only. Parity: bit-identical.

Standalone kernel bench (C=120 trunk shape):
- 360p grid (40x22): +20-21% on both tunes
- 480p grid (53x30): 0-1%
- 720p grid (80x45): +1% base tune, -19% (!) on coc8/slab12

Verdict: real win at small grids, real REGRESSION possible at large ones -
exactly the case for measurement-gated integration. The sg variants joined
tuneConvRB's candidate list (only when the device has 'subgroups'); the
per-(GPU, quality) calibration picks them where they win. Verified: the
tuner selects sg at 360p and base kernels at 480p/720p on the 4060 Ti.
