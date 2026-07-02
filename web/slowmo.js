// Interleave frames with the mids computed between them:
// frames [F0..F_{n-1}], mids [M0..M_{n-2}] -> [F0,M0,F1,M1,...,F_{n-1}] (length 2n-1).
export function assembleSlowmo(frames, mids) {
  const out = [];
  for (let i = 0; i < frames.length; i++) {
    out.push(frames[i]);
    if (i < mids.length) out.push(mids[i]);
  }
  return out;
}
