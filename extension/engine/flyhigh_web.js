/* @ts-self-types="./flyhigh_web.d.ts" */

/**
 * The device the engine runs on; one per page. `await FhEngine.create(options)`.
 */
export class FhEngine {
    static __wrap(ptr) {
        const obj = Object.create(FhEngine.prototype);
        obj.__wbg_ptr = ptr;
        FhEngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FhEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fhengine_free(ptr, 0);
    }
    /**
     * Times the conv kernels of one configuration (5 to 15 s of GPU time; run it idle) and resolves with the table as
     * text for the `tune` session option; an empty string on a device without timestamp queries.
     * @param {FhModel} model
     * @param {any} options
     * @returns {Promise<any>}
     */
    autotune(model, options) {
        _assertClass(model, FhModel);
        const ret = wasm.fhengine_autotune(this.__wbg_ptr, model.__wbg_ptr, options);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Creates the WebGPU device (`{ precision: 'auto' | 'f16' | 'f32' }`; the device needs `shader-f16` for the f16
     * path and falls back to f32 without it).
     * @param {any} options
     * @returns {Promise<FhEngine>}
     */
    static create(options) {
        const ret = wasm.fhengine_create(options);
        return ret;
    }
    /**
     * The `GPUDevice`, to present, sample and copy on the same device.
     * @returns {any}
     */
    device() {
        const ret = wasm.fhengine_device(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * `{ name, class: 'tc-f16' | 'portable-f16' | 'portable-f32', tensor_cores, timestamps, bgra_storage }`
     * @returns {any}
     */
    info() {
        const ret = wasm.fhengine_info(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * A model from its three files: `graph.json` text, the manifest text and the weight bytes.
     * @param {string} graph
     * @param {string} manifest
     * @param {Uint8Array} weights
     * @returns {FhModel}
     */
    loadModel(graph, manifest, weights) {
        const ptr0 = passStringToWasm0(graph, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(manifest, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray8ToWasm0(weights, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.fhengine_loadModel(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return FhModel.__wrap(ret[0]);
    }
    /**
     * @returns {any}
     */
    report() {
        const ret = wasm.fhengine_report(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * The plans of `model` for one frame size.
     * @param {FhModel} model
     * @param {any} options
     * @returns {FhSession}
     */
    session(model, options) {
        _assertClass(model, FhModel);
        const ret = wasm.fhengine_session(this.__wbg_ptr, model.__wbg_ptr, options);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return FhSession.__wrap(ret[0]);
    }
    /**
     * A Stream of `model`: `session` are the session options, `stream` the stream options
     * (`{ mode: 'realtime' | 'video', cadence, sceneCut, heldFrames, inputs, outputs }`, camelCase or snake_case).
     * @param {FhModel} model
     * @param {any} session
     * @param {any} stream
     * @returns {FhStream}
     */
    stream(model, session, stream) {
        _assertClass(model, FhModel);
        const ret = wasm.fhengine_stream(this.__wbg_ptr, model.__wbg_ptr, session, stream);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return FhStream.__wrap(ret[0]);
    }
}
if (Symbol.dispose) FhEngine.prototype[Symbol.dispose] = FhEngine.prototype.free;

export class FhModel {
    static __wrap(ptr) {
        const obj = Object.create(FhModel.prototype);
        obj.__wbg_ptr = ptr;
        FhModelFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FhModelFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fhmodel_free(ptr, 0);
    }
    /**
     * `{ name, train_step, graph_ops, tensors, render_sampling, format_version }`
     * @returns {any}
     */
    info() {
        const ret = wasm.fhmodel_info(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) FhModel.prototype[Symbol.dispose] = FhModel.prototype.free;

/**
 * The scheduler between a video's frames and the display's ticks (ARCHITECTURE.md). Per
 * `requestAnimationFrame`: `tick(now)`, present what it says, generate its requests through the generator and
 * `generated(now, id, outputSlot)` each. Per video frame, once its kind is known: `sourceFrame(now, frame)`,
 * generate the requests it returns, then push the frame.
 */
export class FhPacer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FhPacerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fhpacer_free(ptr, 0);
    }
    /**
     * A measured GPU cost of `stage` ('pair' | 'mid' | 'present'). Returns whether it was taken (not inside the
     * settle window after a mode change).
     * @param {number} now
     * @param {string} stage
     * @param {number} ms
     * @returns {boolean}
     */
    costSample(now, stage, ms) {
        const ptr0 = passStringToWasm0(stage, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.fhpacer_costSample(this.__wbg_ptr, now, ptr0, len0, ms);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * @returns {number}
     */
    delayMs() {
        const ret = wasm.fhpacer_delayMs(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    factor() {
        const ret = wasm.fhpacer_factor(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The host generated (submitted) the frame of request `request` into output slot `output`.
     * @param {number} now
     * @param {number} request
     * @param {number} output
     */
    generated(now, request, output) {
        wasm.fhpacer_generated(this.__wbg_ptr, now, request, output);
    }
    /**
     * @param {number} now
     * @param {number} ms
     */
    hostStall(now, ms) {
        wasm.fhpacer_hostStall(this.__wbg_ptr, now, ms);
    }
    /**
     * Fullscreen, resize: drops are exempt for the settle window.
     * @param {number} now
     */
    modeChange(now) {
        wasm.fhpacer_modeChange(this.__wbg_ptr, now);
    }
    /**
     * `options`: any subset of the Pacer options (`{ mode: 'auto' | 'display_hz' | { factor: 3 } | { target: 120 },
     * fpsLimit, minPresentIntervalMs, ... }`, camelCase or snake_case); unknown keys are rejected with the valid list.
     * @param {any} options
     */
    constructor(options) {
        const ret = wasm.fhpacer_new(options);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        FhPacerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {any}
     */
    options() {
        const ret = wasm.fhpacer_options(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {number}
     */
    penalty() {
        const ret = wasm.fhpacer_penalty(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Records every call from now on into a `trace v1` object named after `host` (a user agent, a page), with the
     * options in force: what a bug report attaches, and what `fh_pacer_replay` and `flyhigh-dev replay` reproduce.
     * @param {string} host
     */
    record(host) {
        const ptr0 = passStringToWasm0(host, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.fhpacer_record(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @returns {any}
     */
    report() {
        const ret = wasm.fhpacer_report(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Seek, new source.
     * @param {number} now
     */
    reset(now) {
        wasm.fhpacer_reset(this.__wbg_ptr, now);
    }
    /**
     * Changes options live; a new mode restarts the controller and the output grid.
     * @param {number} now
     * @param {any} options
     */
    setOptions(now, options) {
        const ret = wasm.fhpacer_setOptions(this.__wbg_ptr, now, options);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The generator answered that the source frame is enough for the request.
     * @param {number} request
     */
    skipped(request) {
        wasm.fhpacer_skipped(this.__wbg_ptr, request);
    }
    /**
     * `frame`: `{ id, ptsMs, kind: 'motion' | 'cut' | 'hold', motion, skipped }`. Returns the requests to generate
     * before the pair changes.
     * @param {number} now
     * @param {any} frame
     * @returns {any}
     */
    sourceFrame(now, frame) {
        const ret = wasm.fhpacer_sourceFrame(this.__wbg_ptr, now, frame);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Ends the recording and returns it (`null` when there was none).
     * @returns {any}
     */
    stopRecording() {
        const ret = wasm.fhpacer_stopRecording(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * `{ present: { frame: { source: id } | { output: id }, due_ms, late_ms, mid, fallback } | null,
     *    generate: [{ id, previous, current, t, due_ms, retime, probe }], factor, dropped }`
     * @param {number} now
     * @returns {any}
     */
    tick(now) {
        const ret = wasm.fhpacer_tick(this.__wbg_ptr, now);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * The recording so far as a plain object, or `null` when not recording.
     * @returns {any}
     */
    trace() {
        const ret = wasm.fhpacer_trace(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) FhPacer.prototype[Symbol.dispose] = FhPacer.prototype.free;

/**
 * Two frames in, one frame at phase `t` out, on textures the session owns.
 */
export class FhSession {
    static __wrap(ptr) {
        const obj = Object.create(FhSession.prototype);
        obj.__wbg_ptr = ptr;
        FhSessionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FhSessionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fhsession_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    addInput() {
        const ret = wasm.fhsession_addInput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    addOutput() {
        const ret = wasm.fhsession_addOutput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} a
     * @param {number} b
     * @param {number} t
     * @param {number} out
     */
    generate(a, b, t, out) {
        const ret = wasm.fhsession_generate(this.__wbg_ptr, a, b, t, out);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The slot's `GPUTexture` (rgba8unorm).
     * @param {number} slot
     * @returns {any}
     */
    inputTexture(slot) {
        const ret = wasm.fhsession_inputTexture(this.__wbg_ptr, slot);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Runs one more generated frame of the last pair and resolves with its GPU milliseconds (`null` without timestamp
     * queries).
     * @param {number} t
     * @param {number} out
     * @returns {Promise<any>}
     */
    midTimed(t, out) {
        const ret = wasm.fhsession_midTimed(this.__wbg_ptr, t, out);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Phase `t` of the last pair into output slot `out`.
     * @param {number} t
     * @param {number} out
     */
    mid(t, out) {
        const ret = wasm.fhsession_mid(this.__wbg_ptr, t, out);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} slot
     * @returns {any}
     */
    outputTexture(slot) {
        const ret = wasm.fhsession_outputTexture(this.__wbg_ptr, slot);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Runs the pair stage of (`a`, `b`) and resolves with its GPU milliseconds (timestamp queries; `null` without
     * them).
     * @param {number} a
     * @param {number} b
     * @returns {Promise<any>}
     */
    pairTimed(a, b) {
        const ret = wasm.fhsession_pairTimed(this.__wbg_ptr, a, b);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * The motion of the interval (`a`, `b`).
     * @param {number} a
     * @param {number} b
     */
    pair(a, b) {
        const ret = wasm.fhsession_pair(this.__wbg_ptr, a, b);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} slot
     * @param {ImageBitmap} b
     */
    pushBitmap(slot, b) {
        const ret = wasm.fhsession_pushBitmap(this.__wbg_ptr, slot, b);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} slot
     * @param {HTMLCanvasElement} c
     */
    pushCanvas(slot, c) {
        const ret = wasm.fhsession_pushCanvas(this.__wbg_ptr, slot, c);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Copies a frame into an input slot (sizes must match). Firefox rejects `<video>` sources: blit through
     * `importExternalTexture` into `inputTexture(slot)` there.
     * @param {number} slot
     * @param {HTMLVideoElement} v
     */
    pushVideo(slot, v) {
        const ret = wasm.fhsession_pushVideo(this.__wbg_ptr, slot, v);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Resolves with the output slot's RGBA8 bytes (a `Uint8Array`).
     * @param {number} out
     * @returns {Promise<any>}
     */
    readRgba(out) {
        const ret = wasm.fhsession_readRgba(this.__wbg_ptr, out);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {any}
     */
    report() {
        const ret = wasm.fhsession_report(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Compiles every kernel the session runs (one pair and one mid); await `device.queue.onSubmittedWorkDone()` after
     * it, before the first real frame.
     */
    warmUp() {
        const ret = wasm.fhsession_warmUp(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The protection mask (`protectionMask`): width * height bytes, 0 interpolates, anything else is taken from the
     * source frames; applies from the next pair.
     * @param {Uint8Array} mask
     */
    writeMask(mask) {
        const ptr0 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.fhsession_writeMask(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * RGBA8 bytes into an input slot.
     * @param {number} slot
     * @param {Uint8Array} rgba
     */
    writeRgba(slot, rgba) {
        const ptr0 = passArray8ToWasm0(rgba, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.fhsession_writeRgba(this.__wbg_ptr, slot, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) FhSession.prototype[Symbol.dispose] = FhSession.prototype.free;

/**
 * Frames with timestamps in, intervals out (ARCHITECTURE.md); the Pacer decides the phases and their times.
 */
export class FhStream {
    static __wrap(ptr) {
        const obj = Object.create(FhStream.prototype);
        obj.__wbg_ptr = ptr;
        FhStreamFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FhStreamFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fhstream_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    addInput() {
        const ret = wasm.fhstream_addInput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    addOutput() {
        const ret = wasm.fhstream_addOutput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Classifies the pair (`previous`, `slot`) before pushing it: resolves with `{ kind: 'motion' | 'cut' | 'hold',
     * mean, max, black, motion }`, or `{ kind: 'motion', busy: true }` when every readback is still in flight. In
     * Firefox submit an empty command buffer every millisecond while it is pending (`device.queue.submit([])`).
     * @param {number} previous
     * @param {number} slot
     * @returns {Promise<any>}
     */
    classify(previous, slot) {
        const ret = wasm.fhstream_classify(this.__wbg_ptr, previous, slot);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * `generate` with its GPU time: `{ written, cost }`, `cost` a promise of milliseconds (`null` without timestamp
     * queries, when nothing was written, or for a cadence composite). The Pacer's `costSample(now, 'mid', ms)` takes it; its
     * probe requests are meant for this call.
     * @param {number} t
     * @param {number} out
     * @returns {any}
     */
    generateTimed(t, out) {
        const ret = wasm.fhstream_generateTimed(this.__wbg_ptr, t, out);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Phase `t` of the current interval into output slot `out`; false = nothing written (t = 0 without re-timing:
     * the source frame is enough).
     * @param {number} t
     * @param {number} out
     * @returns {boolean}
     */
    generate(t, out) {
        const ret = wasm.fhstream_generate(this.__wbg_ptr, t, out);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * @param {number} slot
     * @returns {any}
     */
    inputTexture(slot) {
        const ret = wasm.fhstream_inputTexture(this.__wbg_ptr, slot);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * The slot to write the next source frame into: never one the stream still reads nor one in `inUse` (an array
     * of slots the host still shows); -1 when the ring is exhausted (add inputs).
     * @param {any} in_use
     * @returns {number}
     */
    nextInput(in_use) {
        const ret = wasm.fhstream_nextInput(this.__wbg_ptr, in_use);
        return ret;
    }
    /**
     * @returns {any}
     */
    options() {
        const ret = wasm.fhstream_options(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number} slot
     * @returns {any}
     */
    outputTexture(slot) {
        const ret = wasm.fhstream_outputTexture(this.__wbg_ptr, slot);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number} slot
     * @param {ImageBitmap} b
     */
    pushBitmap(slot, b) {
        const ret = wasm.fhstream_pushBitmap(this.__wbg_ptr, slot, b);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} slot
     * @param {HTMLCanvasElement} c
     */
    pushCanvas(slot, c) {
        const ret = wasm.fhstream_pushCanvas(this.__wbg_ptr, slot, c);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * `push` with the pair's GPU time: the same object plus `cost`, a promise of milliseconds (`null` without
     * timestamp queries, or when no pair ran). The Pacer's `costSample(now, 'pair', ms)` takes it.
     * @param {number} slot
     * @param {number} pts
     * @returns {any}
     */
    pushTimed(slot, pts) {
        const ret = wasm.fhstream_pushTimed(this.__wbg_ptr, slot, pts);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Copies a frame into an input slot. Firefox rejects `<video>` sources: blit through `importExternalTexture`
     * into `inputTexture(slot)` there.
     * @param {number} slot
     * @param {HTMLVideoElement} v
     */
    pushVideo(slot, v) {
        const ret = wasm.fhstream_pushVideo(this.__wbg_ptr, slot, v);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The newest source frame is in `slot` with media timestamp `pts` (ms): runs its pair. Returns
     * `{ ready, baseSlot, statsPending, startPts, endPts }`.
     * @param {number} slot
     * @param {number} pts
     * @returns {any}
     */
    push(slot, pts) {
        const ret = wasm.fhstream_push(this.__wbg_ptr, slot, pts);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Resolves with an output slot's RGBA8 bytes (tests; a player presents the texture).
     * @param {number} out
     * @returns {Promise<any>}
     */
    readRgba(out) {
        const ret = wasm.fhstream_readRgba(this.__wbg_ptr, out);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Whether the current interval's real slot is re-timed (present `generate(0, out)` instead of the source).
     * @returns {boolean}
     */
    realReplaced() {
        const ret = wasm.fhstream_realReplaced(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {any}
     */
    report() {
        const ret = wasm.fhstream_report(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Forget the pushed frames after a seek or a decoder gap.
     */
    reset() {
        wasm.fhstream_reset(this.__wbg_ptr);
    }
    /**
     * Video + cadence: resolves when the pending pair statistics are in (await it after a push with `statsPending`,
     * before the interval's first `generate`).
     * @returns {Promise<any>}
     */
    stats() {
        const ret = wasm.fhstream_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * Compiles every kernel the stream runs (the warm-up pushes and one mid on its own slots, then `reset`); await
     * `device.queue.onSubmittedWorkDone()` after it, before the first real frame.
     */
    warmUp() {
        const ret = wasm.fhstream_warmUp(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The protection mask (`protectionMask` of the session options): width * height bytes, 0 interpolates, anything
     * else is taken from the source frames; applies from the next pair.
     * @param {Uint8Array} mask
     */
    writeMask(mask) {
        const ptr0 = passArray8ToWasm0(mask, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.fhstream_writeMask(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) FhStream.prototype[Symbol.dispose] = FhStream.prototype.free;

/**
 * Replays a `trace v1` object through a fresh Pacer: the report and the per-second table, as `flyhigh-dev replay
 * --json` prints them. Options: the trace's own when `options` is null.
 * @param {any} trace
 * @param {any} options
 * @returns {any}
 */
export function fh_pacer_replay(trace, options) {
    const ret = wasm.fh_pacer_replay(trace, options);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Window_a2a6c4d665047b14: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_2664448a7c667d67: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_4687d8d8c2017d52: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_1f9d30630b8b1d3d: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_e343b7d08827ba72: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_object_3c45d4f2dde4e749: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_90b56bc79aad6f6c: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_8865fb403f8fe9d8: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_number_get_2e0e7dee9f701a71: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_0380ccaa2f57f0d9: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_dcc1a90847f04c41: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_b9a325184985e6ff: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_call_187d372bd5fdd4aa: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_configure_1e2c1c9edad07d26: function() { return handleError(function (arg0, arg1) {
            arg0.configure(arg1);
        }, arguments); },
        __wbg_copyBufferToBuffer_9c174b96fb08d551: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        }, arguments); },
        __wbg_copyExternalImageToTexture_116560f3be856776: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            arg0.copyExternalImageToTexture(arg1, arg2, arg3);
        }, arguments); },
        __wbg_copyTextureToBuffer_1234b3210431ad05: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            arg0.copyTextureToBuffer(arg1, arg2, arg3);
        }, arguments); },
        __wbg_createBindGroup_f539b26ca341308f: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_d800e9b1d41b2ee5: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createCommandEncoder_3352d1ffc36c6fc0: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_224fa2618d9948a0: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createQuerySet_2dc8cde53df9849d: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createQuerySet(arg1);
            return ret;
        }, arguments); },
        __wbg_createSampler_9bd91d7e928c0060: function(arg0, arg1) {
            const ret = arg0.createSampler(arg1);
            return ret;
        },
        __wbg_createShaderModule_cefa51336cb288ae: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_createTexture_ed7e9fc04dd54d84: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createTexture(arg1);
            return ret;
        }, arguments); },
        __wbg_createView_da41c2d2cb212715: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createView(arg1);
            return ret;
        }, arguments); },
        __wbg_description_83b8a393160021b9: function(arg0, arg1) {
            const ret = arg1.description;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_dispatchWorkgroups_56b943172790add0: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_end_82fd0c12ba185009: function(arg0) {
            arg0.end();
        },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_features_dec7bd2fd3d91bd6: function(arg0) {
            const ret = arg0.features;
            return ret;
        },
        __wbg_fhengine_new: function(arg0) {
            const ret = FhEngine.__wrap(arg0);
            return ret;
        },
        __wbg_finish_09ec094c10f41e7b: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_ec1c191f66a895b1: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_from_296ca31f8d0f1c52: function(arg0) {
            const ret = Array.from(arg0);
            return ret;
        },
        __wbg_getBindGroupLayout_349f9101ad6d4d76: function(arg0, arg1) {
            const ret = arg0.getBindGroupLayout(arg1 >>> 0);
            return ret;
        },
        __wbg_getContext_e567868594d2e0c6: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getMappedRange_fb54c6327b2d8d20: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_get_31af05bd4842a84f: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_get_unchecked_288889d017702237: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_gpu_afdd4387c7afe5f9: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_has_eafa12e457ea88fb: function(arg0, arg1, arg2) {
            const ret = arg0.has(getStringFromWasm0(arg1, arg2));
            return ret;
        },
        __wbg_info_971d8b9db3dae69f: function(arg0) {
            const ret = arg0.info;
            return ret;
        },
        __wbg_isArray_e15a2ff68ffdbef2: function(arg0) {
            const ret = Array.isArray(arg0);
            return ret;
        },
        __wbg_isFallbackAdapter_4c8cc3b18677460a: function(arg0) {
            const ret = arg0.isFallbackAdapter;
            return ret;
        },
        __wbg_keys_440be172f17c0265: function(arg0) {
            const ret = Object.keys(arg0);
            return ret;
        },
        __wbg_label_7add8cb37a6ef98f: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_7f3c00c40364105e: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_d4bdea10311bd9cf: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_601ad2e086ef8141: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_mapAsync_b0597127f5037286: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_maxBindGroupsPlusVertexBuffers_52369f089736ef9d: function(arg0) {
            const ret = arg0.maxBindGroupsPlusVertexBuffers;
            return ret;
        },
        __wbg_maxBindGroups_4e424afe6ce86ca2: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_7d035da36821c44f: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_423f4a084e32a195: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_c4cd9126f6d287c6: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_d924670762b9e250: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_707a3868f7cebb59: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_0a4d99463cbd6e5e: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_85123ea0587f7558: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_a3186b4c5267d44f: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_57b297355cfb6204: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_4158f95e673d54c4: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_226b0b70910aa16c: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_0e835fda711fc7e6: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxInterStageShaderVariables_8c4a1d727e2aa35a: function(arg0) {
            const ret = arg0.maxInterStageShaderVariables;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_6675f5e91d9a728a: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_1910fa38a6ed1e1f: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_2e244bded070b18d: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_a285f3ebca51ca0d: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_7aa946f0fc322a2b: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_0e699147ad00502d: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_aabf6add54decfe2: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_dd598b27e9c0c1c4: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_f944266c65dfd1a9: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_59fa6be7cfbeeb53: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_bee5f00a4d706c7f: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_5cf6392c4e9033fe: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_548baa887375d865: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_75d881156591f5da: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_minStorageBufferOffsetAlignment_5ba9b77792bdadb3: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_ab7d52a5293b22bd: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_2156486643462a87: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_c9bce48d4b578c9c: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_617a8cdb8bb1130e: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_a217df4351db4b9b: function() { return handleError(function (arg0, arg1) {
            const ret = new OffscreenCanvas(arg0 >>> 0, arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_new_from_slice_9a868026ffa4208a: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_typed_689a3a281a8d4da6: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_typed_b01cb72a8af741a3: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___js_sys_840f5b3d18ce1e43___Function_fn_wasm_bindgen_c62ffef2a4cf3bda___JsValue_____wasm_bindgen_c62ffef2a4cf3bda___sys__Undefined___js_sys_840f5b3d18ce1e43___Function_fn_wasm_bindgen_c62ffef2a4cf3bda___JsValue_____wasm_bindgen_c62ffef2a4cf3bda___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_byte_offset_and_length_2f5d7fc2a828b74d: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_onSubmittedWorkDone_1190213cee1ecf7e: function(arg0) {
            const ret = arg0.onSubmittedWorkDone();
            return ret;
        },
        __wbg_parse_0fc53dead14b3b42: function() { return handleError(function (arg0, arg1) {
            const ret = JSON.parse(getStringFromWasm0(arg0, arg1));
            return ret;
        }, arguments); },
        __wbg_prototypesetcall_bc27214492979395: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_queueMicrotask_9833f9a49df95a49: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_a72f977e97f23c5f: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_7b62c28143d44293: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_a539af006419f2e9: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_5cb8a582e55d08cb: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolveQuerySet_770f23fabac49845: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.resolveQuerySet(arg1, arg2 >>> 0, arg3 >>> 0, arg4, arg5 >>> 0);
        },
        __wbg_resolve_0076e10020304ede: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_c83391351ce68826: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setPipeline_5146f7b8d2b4af5c: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_set_145a351398b48c65: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_34d08fd992d43d61: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_address_mode_u_a68737cf5d288f95: function(arg0, arg1) {
            arg0.addressModeU = __wbindgen_enum_GpuAddressMode[arg1];
        },
        __wbg_set_address_mode_v_b1c3c45933f540d1: function(arg0, arg1) {
            arg0.addressModeV = __wbindgen_enum_GpuAddressMode[arg1];
        },
        __wbg_set_address_mode_w_889c31cf7022c764: function(arg0, arg1) {
            arg0.addressModeW = __wbindgen_enum_GpuAddressMode[arg1];
        },
        __wbg_set_array_layer_count_22afa0a979e4ad55: function(arg0, arg1) {
            arg0.arrayLayerCount = arg1 >>> 0;
        },
        __wbg_set_aspect_9de5253d4c460895: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_aspect_a48d046965270281: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_aspect_b1a9909bf315433f: function(arg0, arg1) {
            arg0.aspect = __wbindgen_enum_GpuTextureAspect[arg1];
        },
        __wbg_set_base_array_layer_2435ba92c80346ae: function(arg0, arg1) {
            arg0.baseArrayLayer = arg1 >>> 0;
        },
        __wbg_set_base_mip_level_8b6093e875e7c65d: function(arg0, arg1) {
            arg0.baseMipLevel = arg1 >>> 0;
        },
        __wbg_set_beginning_of_pass_write_index_fa9ae10d5d2804ce: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_binding_b6cee57f35ac5190: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_buffer_1548ae88a9188037: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_910a40a90f97cfca: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_bytes_per_row_1e824a5502b54b3d: function(arg0, arg1) {
            arg0.bytesPerRow = arg1 >>> 0;
        },
        __wbg_set_bytes_per_row_c28583f0063160f1: function(arg0, arg1) {
            arg0.bytesPerRow = arg1 >>> 0;
        },
        __wbg_set_code_5d5b0b9e2fd0dca7: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_compare_817cf3695599eaa6: function(arg0, arg1) {
            arg0.compare = __wbindgen_enum_GpuCompareFunction[arg1];
        },
        __wbg_set_compute_bc754d9e37f6e4c9: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_count_d9dc88156fd05bc7: function(arg0, arg1) {
            arg0.count = arg1 >>> 0;
        },
        __wbg_set_depth_or_array_layers_e2f074a0284e4806: function(arg0, arg1) {
            arg0.depthOrArrayLayers = arg1 >>> 0;
        },
        __wbg_set_device_210484a77b675c9c: function(arg0, arg1) {
            arg0.device = arg1;
        },
        __wbg_set_dimension_3da9d03131a9f446: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureDimension[arg1];
        },
        __wbg_set_dimension_56332450afa3e0c0: function(arg0, arg1) {
            arg0.dimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_end_of_pass_write_index_b54a8e3802b9ae0d: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_f26b77ab9548e906: function(arg0, arg1, arg2) {
            arg0.entries = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_entry_point_ac8db535971761fe: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_flip_y_5a9a7c3fd5a2314e: function(arg0, arg1) {
            arg0.flipY = arg1 !== 0;
        },
        __wbg_set_format_283dca56552f07a3: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_66735b94bd868ba2: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_f009e603f7d4c28e: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_height_f6619158e5735877: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_label_09e2da5f8522dbe8: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_17b4858e1eef23dd: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_2fefb39c0e0dbbe8: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_3cb2322e6f6db14c: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_3f2ccaafef5ff7c9: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_6f69e25822616a1b: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_70a09ee68d6b1b26: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_92cd3811e96b487c: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_c3eaf136aa464cba: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_c7987704d29f284b: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_cfe64bca8945ee30: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_ee172cd5f6a96961: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_454e3a091b390cd4: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_gpu_auto_layout_mode_ee432e515c357fa8: function(arg0, arg1) {
            arg0.layout = __wbindgen_enum_GpuAutoLayoutMode[arg1];
        },
        __wbg_set_lod_max_clamp_db24179f67f3aa31: function(arg0, arg1) {
            arg0.lodMaxClamp = arg1;
        },
        __wbg_set_lod_min_clamp_2bbce566e9fefa04: function(arg0, arg1) {
            arg0.lodMinClamp = arg1;
        },
        __wbg_set_mag_filter_db8e6b42d4f8846d: function(arg0, arg1) {
            arg0.magFilter = __wbindgen_enum_GpuFilterMode[arg1];
        },
        __wbg_set_mapped_at_creation_3f320fef6761b02c: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_max_anisotropy_84749fdcec362dc4: function(arg0, arg1) {
            arg0.maxAnisotropy = arg1;
        },
        __wbg_set_min_filter_d435bbfc5a637757: function(arg0, arg1) {
            arg0.minFilter = __wbindgen_enum_GpuFilterMode[arg1];
        },
        __wbg_set_mip_level_03ed7bbd0ded367f: function(arg0, arg1) {
            arg0.mipLevel = arg1 >>> 0;
        },
        __wbg_set_mip_level_count_047936c630acee7b: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_mip_level_count_44bc46a1ae6f6daa: function(arg0, arg1) {
            arg0.mipLevelCount = arg1 >>> 0;
        },
        __wbg_set_mip_level_f3745730372683d5: function(arg0, arg1) {
            arg0.mipLevel = arg1 >>> 0;
        },
        __wbg_set_mipmap_filter_62fb49a84b0747ff: function(arg0, arg1) {
            arg0.mipmapFilter = __wbindgen_enum_GpuMipmapFilterMode[arg1];
        },
        __wbg_set_module_c9946af218d23b53: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_offset_f64_127e8a0aa5c5485a: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_offset_f64_a903425d5a8e5815: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_offset_f64_d1d115dd438165b5: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_origin_gpu_origin_2d_dict_8156da81552cd590: function(arg0, arg1) {
            arg0.origin = arg1;
        },
        __wbg_set_origin_gpu_origin_3d_dict_0619d4860adb4eb6: function(arg0, arg1) {
            arg0.origin = arg1;
        },
        __wbg_set_origin_gpu_origin_3d_dict_abd0dec737979786: function(arg0, arg1) {
            arg0.origin = arg1;
        },
        __wbg_set_power_preference_b42d00a8facfbade: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_premultiplied_alpha_6d23532454ce9a12: function(arg0, arg1) {
            arg0.premultipliedAlpha = arg1 !== 0;
        },
        __wbg_set_query_set_f9a94586851fdba2: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_required_features_bbab71414c45e621: function(arg0, arg1, arg2) {
            arg0.requiredFeatures = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_required_limits_837f62d865e7cfac: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resource_8fd8658b30d86ecf: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_buffer_binding_33099b25da65b610: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_texture_view_4cffe7bc7c8e5cbe: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_rows_per_image_c6d50d227e634379: function(arg0, arg1) {
            arg0.rowsPerImage = arg1 >>> 0;
        },
        __wbg_set_rows_per_image_deb456502f23c260: function(arg0, arg1) {
            arg0.rowsPerImage = arg1 >>> 0;
        },
        __wbg_set_sample_count_481c255a12054e1d: function(arg0, arg1) {
            arg0.sampleCount = arg1 >>> 0;
        },
        __wbg_set_size_f64_2f591b0654540477: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_f64_e844c985b8f95261: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_gpu_extent_3d_dict_adf57388ab1d4f18: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_source_211aa06496d82b73: function(arg0, arg1) {
            arg0.source = arg1;
        },
        __wbg_set_source_html_canvas_element_5577621adc0f8ff6: function(arg0, arg1) {
            arg0.source = arg1;
        },
        __wbg_set_source_html_image_element_23fa06e99ca07cc0: function(arg0, arg1) {
            arg0.source = arg1;
        },
        __wbg_set_source_html_video_element_e57b5cacb7b02736: function(arg0, arg1) {
            arg0.source = arg1;
        },
        __wbg_set_source_image_data_c80600573a809775: function(arg0, arg1) {
            arg0.source = arg1;
        },
        __wbg_set_texture_4e0ee4c98ac25b81: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_texture_95f2bfdf7767e76f: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_5f240bbaad97fcaf: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_type_7f7e54057b801caa: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuQueryType[arg1];
        },
        __wbg_set_usage_1ee33d98267e787d: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_d53ee6f0c7aedbfa: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_usage_f3e34822998d2147: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_view_formats_427069064d8b7139: function(arg0, arg1, arg2) {
            arg0.viewFormats = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_width_b20525f5f4df4eb8: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_x_58965df5af1ad27d: function(arg0, arg1) {
            arg0.x = arg1 >>> 0;
        },
        __wbg_set_x_f470b03dd54724cd: function(arg0, arg1) {
            arg0.x = arg1 >>> 0;
        },
        __wbg_set_y_4c44eb40ebca5bfc: function(arg0, arg1) {
            arg0.y = arg1 >>> 0;
        },
        __wbg_set_y_b79cba22b145de99: function(arg0, arg1) {
            arg0.y = arg1 >>> 0;
        },
        __wbg_set_z_2e6820ef0f5821ed: function(arg0, arg1) {
            arg0.z = arg1 >>> 0;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_266715b9d96ba635: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_10fb7dc1ae063179: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_0b583911f537483a: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_d7f903d1508cbdc4: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_stringify_52ff602c1cc4fbb6: function() { return handleError(function (arg0) {
            const ret = JSON.stringify(arg0);
            return ret;
        }, arguments); },
        __wbg_subgroupMaxSize_b43be0aa16182403: function(arg0) {
            const ret = arg0.subgroupMaxSize;
            return ret;
        },
        __wbg_subgroupMinSize_03feb6ee0cda6775: function(arg0) {
            const ret = arg0.subgroupMinSize;
            return ret;
        },
        __wbg_submit_077c85cc28e36892: function(arg0, arg1, arg2) {
            arg0.submit(getArrayJsValueViewFromWasm0(arg1, arg2));
        },
        __wbg_then_c949d5a25a4e78f8: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_e71170d78fcf8954: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_unconfigure_835307f58dc68d80: function(arg0) {
            arg0.unconfigure();
        },
        __wbg_unmap_6a96b14c9ef5f7f5: function(arg0) {
            arg0.unmap();
        },
        __wbg_writeBuffer_f4bb3f54adfe1330: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.writeBuffer(arg1, arg2, getArrayU8FromWasm0(arg3, arg4), arg5, arg6);
        }, arguments); },
        __wbg_writeTexture_30e592e8c061c3d9: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.writeTexture(arg1, getArrayU8FromWasm0(arg2, arg3), arg4, arg5);
        }, arguments); },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 32, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true_);
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUDevice")], shim_idx: 32, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__68);
            return ret;
        },
        __wbindgen_generic_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("any")], shim_idx: 32, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__69);
            return ret;
        },
        __wbindgen_generic_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("undefined")], shim_idx: 32, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__70);
            return ret;
        },
        __wbindgen_generic_0000000000000005: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_0000000000000007: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./flyhigh_web_bg.js": import0,
    };
}

function wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__68(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__68(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__69(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__69(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__70(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___wasm_bindgen_c62ffef2a4cf3bda___sys__JsNullable_wgpu_3e8ff71263307194___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_608f92abc48d28da___result__Result_____wasm_bindgen_c62ffef2a4cf3bda___JsError___true__70(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___js_sys_840f5b3d18ce1e43___Function_fn_wasm_bindgen_c62ffef2a4cf3bda___JsValue_____wasm_bindgen_c62ffef2a4cf3bda___sys__Undefined___js_sys_840f5b3d18ce1e43___Function_fn_wasm_bindgen_c62ffef2a4cf3bda___JsValue_____wasm_bindgen_c62ffef2a4cf3bda___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_c62ffef2a4cf3bda___convert__closures_____invoke___js_sys_840f5b3d18ce1e43___Function_fn_wasm_bindgen_c62ffef2a4cf3bda___JsValue_____wasm_bindgen_c62ffef2a4cf3bda___sys__Undefined___js_sys_840f5b3d18ce1e43___Function_fn_wasm_bindgen_c62ffef2a4cf3bda___JsValue_____wasm_bindgen_c62ffef2a4cf3bda___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuAddressMode = ["clamp-to-edge", "repeat", "mirror-repeat"];


const __wbindgen_enum_GpuAutoLayoutMode = ["auto"];


const __wbindgen_enum_GpuCompareFunction = ["never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"];


const __wbindgen_enum_GpuFilterMode = ["nearest", "linear"];


const __wbindgen_enum_GpuMipmapFilterMode = ["nearest", "linear"];


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];


const __wbindgen_enum_GpuQueryType = ["occlusion", "timestamp"];


const __wbindgen_enum_GpuTextureAspect = ["all", "stencil-only", "depth-only"];


const __wbindgen_enum_GpuTextureDimension = ["1d", "2d", "3d"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16unorm", "r16snorm", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16unorm", "rg16snorm", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16unorm", "rgba16snorm", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];


const __wbindgen_enum_GpuTextureViewDimension = ["1d", "2d", "2d-array", "cube", "cube-array", "3d"];
const FhEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fhengine_free(ptr, 1));
const FhModelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fhmodel_free(ptr, 1));
const FhPacerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fhpacer_free(ptr, 1));
const FhSessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fhsession_free(ptr, 1));
const FhStreamFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fhstream_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayJsValueViewFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    return result;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('flyhigh_web_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
