/** Kernel autotune result for the register-blocked conv (per GPU + shape). */
export interface ConvTune {
  coc: number;
  slab: number;
  sg?: boolean;
  ms?: number;
}

export interface CreateRTOptions {
  /** Output width in pixels. Must be divisible by 16. */
  w: number;
  /** Output height in pixels. Must be divisible by 16. */
  h: number;
  /** Raw weights blob (the released .bin file). */
  weightsBin: ArrayBuffer;
  /** Weights manifest (the released .json file, parsed). */
  weightsManifest: Record<string, { offset: number; shape: number[] }>;
  /** Kernel tune from tuneConvRB; omit for safe defaults. */
  convTune?: ConvTune | null;
  /** Inputs are GPUTextures instead of RGBA8 CPU buffers. */
  textureInput?: boolean;
  /** Mids are written into GPUTextures instead of read back to CPU. */
  textureOutput?: boolean;
  /** Bake the static-region guard into the flow kernel. */
  staticGuard?: boolean;
}

export interface RT {
  /** Buffer mode only: interpolate one mid between two RGBA8 frames. */
  run(rgbaA: Uint8Array, rgbaB: Uint8Array, t?: number): Promise<Uint8Array>;
  /**
   * Batched mids for every t. Buffer mode returns RGBA8 arrays; texture mode
   * writes into outTexs and resolves null (nothing crosses the bus).
   */
  runMulti(
    a: Uint8Array | GPUTexture,
    b: Uint8Array | GPUTexture,
    ts: number[],
    outTexs?: GPUTexture[],
  ): Promise<Uint8Array[] | null>;
  /** Texture mode: run the t-free trunk once for a frame pair. */
  prepPair(a: GPUTexture, b: GPUTexture): void;
  /** Texture mode: one mid at timestep t into outTex (call prepPair first). */
  runT(t: number, outTex: GPUTexture): void;
  /** Per-pass GPU timings (requires timestamp-query). */
  profile(rgbaA: Uint8Array, rgbaB: Uint8Array): Promise<string>;
  readonly w: number;
  readonly h: number;
}

export function createRT(device: GPUDevice, opts: CreateRTOptions): Promise<RT>;

/** Bench conv kernel variants on the real shape; persist the result per GPU. */
export function tuneConvRB(
  device: GPUDevice,
  shape: { ci: number; co: number; w16: number; h16: number },
): Promise<ConvTune>;
