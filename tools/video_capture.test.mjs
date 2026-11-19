import assert from 'node:assert/strict';
import test from 'node:test';
import { VideoSource } from '../extension/engine/adapters.js';

test('video capture uses an external texture even when the video and input slot match', () => {
  const video = { videoWidth: 1920, videoHeight: 1080 };
  const external = {};
  const view = {};
  const input = { width: 1920, height: 1080, createView: () => view };
  const stream = {
    inputTexture(slot) { assert.equal(slot, 0); return input; },
    pushVideo() { assert.fail('direct video copies can corrupt chroma'); },
  };
  const pipeline = { getBindGroupLayout: () => ({}) };
  const commands = {};
  const pass = {
    setPipeline(value) { assert.equal(value, pipeline); },
    setBindGroup() {},
    draw(vertices) { assert.equal(vertices, 3); },
    end() {},
  };
  let submits = 0;
  const device = {
    importExternalTexture({ source }) { assert.equal(source, video); return external; },
    createSampler: () => ({}),
    createShaderModule: ({ code }) => {
      assert.match(code, /textureSampleBaseClampToEdge/);
      return {};
    },
    createRenderPipeline: () => pipeline,
    createBindGroup: ({ entries }) => {
      assert.equal(entries[0].resource, external);
      return {};
    },
    createCommandEncoder: () => ({
      beginRenderPass({ colorAttachments }) {
        assert.equal(colorAttachments[0].view, view);
        return pass;
      },
      finish: () => commands,
    }),
    queue: {
      submit(batch) { assert.deepEqual(batch, [commands]); submits++; },
    },
  };
  const source = new VideoSource(device, stream);
  source.write(0, video);
  assert.equal(source.path, 'external');
  assert.equal(submits, 1);
});
