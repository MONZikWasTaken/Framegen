// IFRNet Vimeo90K inference graph for Framegen's WebGPU path.
//
// This intentionally uses f32 activations first. It is a correctness-oriented
// high-quality tier; the v7 runtime remains the low-latency default. The graph
// mirrors IFRNet/models/IFRNet.py exactly: mean normalization, two encoders,
// four decoder stages, border grid-sampling, and final residual blending.

import { wgslConv } from './rt.js';

const WG = 8;
const gx = n => Math.ceil(n / WG);
const elements = (c, w, h) => c * w * h;

function wgslPairPrep(W, H) {
  return /* wgsl */`
@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> imgs: array<f32>; // BGR, [2,3,H,W]
@group(0) @binding(4) var<storage, read_write> sums: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y); let P = ${W * H};
  var s = 0.0;
  if (x < ${W} && y < ${H}) {
    let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
    let a = textureSampleLevel(tex0, samp, uv, 0.0).bgr;
    let b = textureSampleLevel(tex1, samp, uv, 0.0).bgr;
    let o = y * ${W} + x;
    imgs[o] = a.x; imgs[P + o] = a.y; imgs[2 * P + o] = a.z;
    imgs[3 * P + o] = b.x; imgs[4 * P + o] = b.y; imgs[5 * P + o] = b.z;
    s = a.x + a.y + a.z + b.x + b.y + b.z;
  }
  partial[li] = s; workgroupBarrier();
  var n = 32u;
  loop { if (li < n) { partial[li] += partial[li + n]; } workgroupBarrier(); if (n == 1u) { break; } n /= 2u; }
  if (li == 0u) { sums[wid.y * ${Math.ceil(W / WG)} + wid.x] = partial[0]; }
}`;
}

function wgslMean(groups, denominator) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> sums: array<f32>;
@group(0) @binding(1) var<storage, read_write> mean: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var v = 0.0; var i = li;
  loop { if (i >= ${groups}u) { break; } v += sums[i]; i += 256u; }
  part[li] = v; workgroupBarrier();
  var n = 128u;
  loop { if (li < n) { part[li] += part[li + n]; } workgroupBarrier(); if (n == 1u) { break; } n /= 2u; }
  if (li == 0u) { mean[0] = part[0] / ${denominator}.0; }
}`;
}

function wgslCenter(C, W, H) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> imgs: array<f32>;
@group(0) @binding(1) var<storage, read> mean: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i < ${C * W * H}u) { dst[i] = imgs[i] - mean[0]; }
}`;
}

function wgslCopySide(C, W, H, side) {
  const P = W * H;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> sideSrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= ${C * P}u) { return; }
  let c = i / ${P}u;
  dst[i] = select(src[i], sideSrc[(c - ${C - side}u) * ${P}u + i % ${P}u], c >= ${C - side}u);
}`;
}

function wgslAddPrelu(C, W, H) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> alpha: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= ${C * W * H}u) { return; }
  let v = a[i] + b[i]; let c = i / ${W * H}u;
  dst[i] = select(alpha[c] * v, v, v >= 0.0);
}`;
}

function wgslDecoderInput(F, E, W, H) {
  const P = W * H;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> ft: array<f32>;
@group(0) @binding(1) var<storage, read> f0: array<f32>;
@group(0) @binding(2) var<storage, read> f1: array<f32>;
@group(0) @binding(3) var<storage, read> flow: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
fn tap(src: ptr<storage, array<f32>, read>, c: i32, x: i32, y: i32) -> f32 {
  return (*src)[c * ${P} + clamp(y, 0, ${H - 1}) * ${W} + clamp(x, 0, ${W - 1})];
}
fn warp(src: ptr<storage, array<f32>, read>, c: i32, x: f32, y: f32) -> f32 {
  let x0 = i32(floor(x)); let y0 = i32(floor(y)); let fx = x - f32(x0); let fy = y - f32(y0);
  return mix(mix(tap(src,c,x0,y0),tap(src,c,x0+1,y0),fx),mix(tap(src,c,x0,y0+1),tap(src,c,x0+1,y0+1),fx),fy);
}
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y); let c = i32(gid.z); if (x >= ${W} || y >= ${H}) { return; }
  let o = y * ${W} + x; let P = ${P};
  if (c < ${F}) { dst[c * P + o] = ft[c * P + o]; return; }
  let fc = c - ${F};
  if (fc < ${E}) { dst[c * P + o] = warp(&f0, fc, f32(x) + flow[o], f32(y) + flow[P + o]); return; }
  if (fc < ${2 * E}) { let q = fc - ${E}; dst[c * P + o] = warp(&f1, q, f32(x) + flow[2 * P + o], f32(y) + flow[3 * P + o]); return; }
  dst[c * P + o] = flow[(fc - ${2 * E}) * P + o];
}`;
}

function wgslDecoder4Input(W, H) {
  const P = W * H;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> f0: array<f32>;
@group(0) @binding(1) var<storage, read> f1: array<f32>;
@group(0) @binding(2) var<storage, read> time: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x=i32(gid.x); let y=i32(gid.y); let c=i32(gid.z); if(x>=${W}||y>=${H}) { return; }
  let o=y*${W}+x; if(c<96) { dst[c*${P}+o]=f0[c*${P}+o]; } else if(c<192) { dst[c*${P}+o]=f1[(c-96)*${P}+o]; } else { dst[c*${P}+o]=time[0]; }
}`;
}

function wgslFlowUpdate(W, H) {
  const OW = W * 2, OH = H * 2, P = W * H, OP = OW * OH;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> prev: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
fn tap(c:i32,x:i32,y:i32)->f32{return prev[c*${P}+clamp(y,0,${H-1})*${W}+clamp(x,0,${W-1})];}
@compute @workgroup_size(${WG},${WG})
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let x=i32(gid.x);let y=i32(gid.y);let c=i32(gid.z);if(x>=${OW}||y>=${OH}){return;}let sx=(f32(x)+0.5)/2.0-0.5;let sy=(f32(y)+0.5)/2.0-0.5;let x0=i32(floor(sx));let y0=i32(floor(sy));let fx=sx-f32(x0);let fy=sy-f32(y0);let up=mix(mix(tap(c,x0,y0),tap(c,x0+1,y0),fx),mix(tap(c,x0,y0+1),tap(c,x0+1,y0+1),fx),fy);let o=y*${OW}+x;dst[c*${OP}+o]=delta[c*${OP}+o]+2.0*up;}`;
}

function wgslCopy(C, W, H) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i < ${C * W * H}u) { dst[i] = src[i]; }
}`;
}

function wgslCopyRange(C, W, H, start) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i < ${C * W * H}u) { let c = i / ${W * H}u; dst[i] = src[(c + ${start}u) * ${W * H}u + i % ${W * H}u]; }
}`;
}

// Unlike the RIFE head (five output channels), IFRNet's decoder transposes up
// to 76 channels. Keep this shader channel-generic: unrolling all 76
// accumulators creates multi-minute pipeline compilation on some drivers.
function wgslDeconv(CI, CO, IW, IH) {
  const OW = IW * 2, OH = IH * 2;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>; // [CI,CO,4,4]
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x=i32(gid.x); let y=i32(gid.y); let co=i32(gid.z);
  if (x>=${OW} || y>=${OH} || co>=${CO}) { return; }
  let py=(y+1)&1; let px=(x+1)&1; var a=bias[co];
  for (var ky=py; ky<4; ky+=2) { let ty=y+1-ky; let iy=ty/2; if (ty<0 || iy>=${IH}) { continue; }
    for (var kx=px; kx<4; kx+=2) { let tx=x+1-kx; let ix=tx/2; if (tx<0 || ix>=${IW}) { continue; }
      for (var ci=0; ci<${CI}; ci++) { a += src[ci*${IW * IH}+iy*${IW}+ix] * weight[(ci*${CO}+co)*16+ky*4+kx]; }
    }
  }
  dst[co*${OW * OH}+y*${OW}+x]=a;
}`;
}

function wgslOutput(W, H) {
  const HP = W * H;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> out1: array<f32>;
@group(0) @binding(1) var<storage, read> flow: array<f32>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0) var tex0: texture_2d<f32>;
@group(1) @binding(1) var tex1: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;
fn tapOut(c:i32,x:i32,y:i32)->f32{return out1[c*${HP}+clamp(y,0,${H-1})*${W}+clamp(x,0,${W-1})];}
fn tapFlow(c:i32,x:i32,y:i32)->f32{return flow[c*${HP}+clamp(y,0,${H-1})*${W}+clamp(x,0,${W-1})];}
fn up(c:i32,mx:f32,my:f32)->f32 { let x0=i32(floor(mx));let y0=i32(floor(my));let fx=mx-f32(x0);let fy=my-f32(y0);return mix(mix(tapOut(c,x0,y0),tapOut(c,x0+1,y0),fx),mix(tapOut(c,x0,y0+1),tapOut(c,x0+1,y0+1),fx),fy); }
fn upFlow(c:i32,mx:f32,my:f32)->f32 { let x0=i32(floor(mx));let y0=i32(floor(my));let fx=mx-f32(x0);let fy=my-f32(y0);return mix(mix(tapFlow(c,x0,y0),tapFlow(c,x0+1,y0),fx),mix(tapFlow(c,x0,y0+1),tapFlow(c,x0+1,y0+1),fx),fy); }
fn warp(t:texture_2d<f32>,x:f32,y:f32)->vec3<f32>{return textureSampleLevel(t,samp,(vec2<f32>(x,y)+0.5)/vec2<f32>(textureDimensions(t)),0.0).bgr;}
@compute @workgroup_size(${WG},${WG})
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let x=i32(gid.x);let y=i32(gid.y);let d=vec2<f32>(textureDimensions(outTex));if(f32(x)>=d.x||f32(y)>=d.y){return;}let mx=(f32(x)+0.5)*${W}.0/d.x-0.5;let my=(f32(y)+0.5)*${H}.0/d.y-0.5;let scale=vec2<f32>(d.x/${W}.0,d.y/${H}.0);let f0=vec2<f32>(upFlow(0,mx,my),upFlow(1,mx,my))*scale;let f1=vec2<f32>(upFlow(2,mx,my),upFlow(3,mx,my))*scale;let p=(vec2<f32>(f32(x),f32(y))+0.5);let m=1.0/(1.0+exp(-up(4,mx,my)));let bgr=clamp(warp(tex0,p.x+f0.x,p.y+f0.y)*m+warp(tex1,p.x+f1.x,p.y+f1.y)*(1.0-m)+vec3<f32>(up(5,mx,my),up(6,mx,my),up(7,mx,my)),vec3<f32>(0.0),vec3<f32>(1.0));textureStore(outTex,vec2<i32>(x,y),vec4<f32>(bgr.z,bgr.y,bgr.x,1.0));}`;
}

export async function createIFRNetRT(device, { w, h, weightsBin, weightsManifest }) {
  if (w % 16 || h % 16) throw new Error(`ifrnet: dimensions must be divisible by 16 (got ${w}x${h})`);
  if (w > 1280 || h > 720) throw new Error('ifrnet: experimental runtime currently supports up to 720p internal resolution');
  const man = weightsManifest.tensors || weightsManifest;
  if (weightsManifest.format && weightsManifest.format !== 'framegen-ifrnet-v1') throw new Error('ifrnet: unsupported weight format');
  const required = 'encoder.pyramid1.0.0.weight';
  if (!man[required]) throw new Error('ifrnet: missing official IFRNet tensors');
  const all = [];
  const make = (n, usage = GPUBufferUsage.STORAGE) => { const b=device.createBuffer({size:Math.ceil(n*4/4)*4,usage:usage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});all.push(b);return b; };
  const view = (b, offset = 0, size) => ({ buffer:b, offset, ...(size ? { size } : {}) });
  const pipe = code => device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code}),entryPoint:'main'}});
  const bind = (p, resources) => device.createBindGroup({layout:p.getBindGroupLayout(0),entries:resources.map((resource,binding)=>({binding,resource}))});
  const weights = {};
  for (const [name, meta] of Object.entries(man)) { const n=meta.shape.reduce((a,b)=>a*b,1); const b=make(n); device.queue.writeBuffer(b,0,weightsBin,meta.offset*4,n*4); weights[name]=b; }
  const ones = {};
  const alpha = (name, n, identity=false) => { if (!identity) return weights[name]; if (!ones[n]) { ones[n]=make(n); device.queue.writeBuffer(ones[n],0,new Float32Array(n).fill(1)); } return ones[n]; };
  const buf = (c,W,H) => make(elements(c,W,H));
  const P = w*h, W1=w/2,H1=h/2,W2=w/4,H2=h/4,W3=w/8,H3=h/8,W4=w/16,H4=h/16;
  const imgs=buf(6,w,h), centered=buf(6,w,h), partial=make(gx(w)*gx(h)), mean=make(1), time=make(1);
  const f0_1=buf(32,W1,H1),f1_1=buf(32,W1,H1),f0_2=buf(48,W2,H2),f1_2=buf(48,W2,H2),f0_3=buf(72,W3,H3),f1_3=buf(72,W3,H3),f0_4=buf(96,W4,H4),f1_4=buf(96,W4,H4);
  const stages=[{C:192,side:32,W:W4,H:H4},{C:216,side:32,W:W3,H:H3},{C:144,side:32,W:W2,H:H2},{C:96,side:32,W:W1,H:H1}].map(s=>({...s,pre:buf(s.C,s.W,s.H),a:buf(s.C,s.W,s.H),b:buf(s.C,s.W,s.H),sideIn:buf(s.side,s.W,s.H),sideBuf:buf(s.side,s.W,s.H)}));
  const decIn4=buf(193,W4,H4), decIn3=buf(220,W3,H3), decIn2=buf(148,W2,H2), decIn1=buf(100,W1,H1);
  const out4=buf(76,W3,H3),out3=buf(52,W2,H2),out2=buf(36,W1,H1),final=buf(8,w,h);
  const flow3=buf(4,W3,H3),flow2=buf(4,W2,H2),flow1=buf(4,W1,H1),flow0=buf(4,w,h);
  const sampler=device.createSampler({magFilter:'linear',minFilter:'linear',addressModeU:'clamp-to-edge',addressModeV:'clamp-to-edge'});
  const prepP=pipe(wgslPairPrep(w,h)), meanP=pipe(wgslMean(gx(w)*gx(h),6*w*h));
  const centerP=pipe(wgslCenter(6,w,h));
  let current=null;
  const prepBg = (a,b) => bind(prepP,[a.createView(),b.createView(),sampler,imgs,partial]);
  const meanBg=bind(meanP,[partial,mean]); const centerBg=bind(centerP,[imgs,mean,centered]);
  function conv(name, src, dst, CI, CO, IW, IH, stride=1, identity=false) {
    const OW=IW/stride,OH=IH/stride, p=pipe(wgslConv(CI,CO,IW,IH,OW,OH,stride,false,false));
    const base=name.replace(/\.weight$/,'');
    const prelu=name.replace(/\.0\.weight$/, '.1.weight');
    return {p,g:bind(p,[src,weights[name],weights[base+'.bias'],alpha(prelu,CO,identity),dst]),d:[gx(OW),gx(OH),CO/4]};
  }
  function deconv(name,src,dst,CI,CO,IW,IH){const p=pipe(wgslDeconv(CI,CO,IW,IH));const base=name.replace(/\.weight$/,'');return {p,g:bind(p,[src,weights[name],weights[base+'.bias'],dst]),d:[gx(IW*2),gx(IH*2),CO]};}
  const enc=[];
  for (const [prefix, input, fs] of [['f0', null,[f0_1,f0_2,f0_3,f0_4]],['f1', null,[f1_1,f1_2,f1_3,f1_4]]]) {
    let src = input; const base = prefix==='f0' ? 0 : 3;
    const dims=[[3,32,w,h,W1,H1,2],[32,32,W1,H1,W1,H1,1],[32,48,W1,H1,W2,H2,2],[48,48,W2,H2,W2,H2,1],[48,72,W2,H2,W3,H3,2],[72,72,W3,H3,W3,H3,1],[72,96,W3,H3,W4,H4,2],[96,96,W4,H4,W4,H4,1]];
    const names=['encoder.pyramid1.0.0.weight','encoder.pyramid1.1.0.weight','encoder.pyramid2.0.0.weight','encoder.pyramid2.1.0.weight','encoder.pyramid3.0.0.weight','encoder.pyramid3.1.0.weight','encoder.pyramid4.0.0.weight','encoder.pyramid4.1.0.weight'];
    const tmp=[buf(32,W1,H1),buf(48,W2,H2),buf(72,W3,H3),buf(96,W4,H4)];
    const outs=[tmp[0],fs[0],tmp[1],fs[1],tmp[2],fs[2],tmp[3],fs[3]];
    for(let i=0;i<8;i++){const [ci,co,iw,ih,,,st]=dims[i]; const source=i===0?(base===0 ? view(centered,0,3*P*4) : view(centered,3*P*4,3*P*4)):outs[i-1]; enc.push(conv(names[i],source,outs[i],ci,co,iw,ih,st));}
  }
  function dispatch(pass, op){pass.setPipeline(op.p);pass.setBindGroup(0,op.g);pass.dispatchWorkgroups(...op.d);}
  function resblock(stage, id, input, output){
    const b=`decoder${id}.convblock.1.`; const {C,side,W,H,a,b:work,sideIn,sideBuf}=stage;
    const c1=conv(b+'conv1.0.weight',input,a,C,C,W,H); const extract1=pipe(wgslCopyRange(side,W,H,C-side)),extract1g=bind(extract1,[a,sideIn]); const cs1=conv(b+'conv2.0.weight',sideIn,sideBuf,side,side,W,H);
    const copy1=pipe(wgslCopySide(C,W,H,side)),copy1g=bind(copy1,[a,sideBuf,work]);
    const c3=conv(b+'conv3.0.weight',work,a,C,C,W,H); const extract2=pipe(wgslCopyRange(side,W,H,C-side)),extract2g=bind(extract2,[a,sideIn]); const cs2=conv(b+'conv4.0.weight',sideIn,sideBuf,side,side,W,H);
    const copy2=pipe(wgslCopySide(C,W,H,side)),copy2g=bind(copy2,[a,sideBuf,work]);
    const c5=conv(b+'conv5.weight',work,a,C,C,W,H,1,true); const add=pipe(wgslAddPrelu(C,W,H)),addg=bind(add,[input,a,weights[b+'prelu.weight'],output]);
    return [c1,{p:extract1,g:extract1g,d:[Math.ceil(side*W*H/256),1,1]},cs1,{p:copy1,g:copy1g,d:[Math.ceil(C*W*H/256),1,1]},c3,{p:extract2,g:extract2g,d:[Math.ceil(side*W*H/256),1,1]},cs2,{p:copy2,g:copy2g,d:[Math.ceil(C*W*H/256),1,1]},c5,{p:add,g:addg,d:[Math.ceil(C*W*H/256),1,1]}];
  }
  const ft3=buf(72,W3,H3),ft2=buf(48,W2,H2),ft1=buf(32,W1,H1);
  const d4inP=pipe(wgslDecoder4Input(W4,H4)),d4inG=bind(d4inP,[f0_4,f1_4,time,decIn4]);
  const di3P=pipe(wgslDecoderInput(72,72,W3,H3)),di3G=bind(di3P,[ft3,f0_3,f1_3,flow3,decIn3]);
  const di2P=pipe(wgslDecoderInput(48,48,W2,H2)),di2G=bind(di2P,[ft2,f0_2,f1_2,flow2,decIn2]);
  const di1P=pipe(wgslDecoderInput(32,32,W1,H1)),di1G=bind(di1P,[ft1,f0_1,f1_1,flow1,decIn1]);
  const first=[conv('decoder4.convblock.0.0.weight',decIn4,stages[0].pre,193,192,W4,H4),conv('decoder3.convblock.0.0.weight',decIn3,stages[1].pre,220,216,W3,H3),conv('decoder2.convblock.0.0.weight',decIn2,stages[2].pre,148,144,W2,H2),conv('decoder1.convblock.0.0.weight',decIn1,stages[3].pre,100,96,W1,H1)];
  const rb=[resblock(stages[0],4,stages[0].pre,stages[0].b),resblock(stages[1],3,stages[1].pre,stages[1].b),resblock(stages[2],2,stages[2].pre,stages[2].b),resblock(stages[3],1,stages[3].pre,stages[3].b)];
  const dc=[deconv('decoder4.convblock.2.weight',stages[0].b,out4,192,76,W4,H4),deconv('decoder3.convblock.2.weight',stages[1].b,out3,216,52,W3,H3),deconv('decoder2.convblock.2.weight',stages[2].b,out2,144,36,W2,H2),deconv('decoder1.convblock.2.weight',stages[3].b,final,96,8,W1,H1)];
  const copy4P=pipe(wgslCopy(4,W3,H3)),copy4G=bind(copy4P,[out4,flow3]);
  const ft3P=pipe(wgslCopyRange(72,W3,H3,4)),ft3G=bind(ft3P,[out4,ft3]);
  const ft2P=pipe(wgslCopyRange(48,W2,H2,4)),ft2G=bind(ft2P,[out3,ft2]);
  const ft1P=pipe(wgslCopyRange(32,W1,H1,4)),ft1G=bind(ft1P,[out2,ft1]);
  const fu3=pipe(wgslFlowUpdate(W4,H4)),fu3g=bind(fu3,[out4,out3,flow3]);
  const fu2=pipe(wgslFlowUpdate(W3,H3)),fu2g=bind(fu2,[flow3,out2,flow2]);
  const fu1=pipe(wgslFlowUpdate(W1,H1)),fu1g=bind(fu1,[flow1,final,flow0]);
  const outP=pipe(wgslOutput(w,h));
  function submit(p,g,d,group1) { const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,g);if(group1)pass.setBindGroup(1,group1);pass.dispatchWorkgroups(...d);pass.end();device.queue.submit([encoder.finish()]); }
  function prepPair(a,b){ current={a,b}; submit(prepP,prepBg(a,b),[gx(w),gx(h),1]);submit(meanP,meanBg,[1,1,1]);submit(centerP,centerBg,[Math.ceil(6*P/256),1,1]);for(const op of enc)submit(op.p,op.g,op.d); }
  function runT(t,outTex){if(!current)throw new Error('ifrnet: runT before prepPair');device.queue.writeBuffer(time,0,new Float32Array([t]));const go=(p,g,d)=>submit(p,g,d);go(d4inP,d4inG,[gx(W4),gx(H4),193]);go(first[0].p,first[0].g,first[0].d);for(const op of rb[0])go(op.p,op.g,op.d);go(dc[0].p,dc[0].g,dc[0].d);go(copy4P,copy4G,[Math.ceil(4*W3*H3/256),1,1]);go(ft3P,ft3G,[Math.ceil(72*W3*H3/256),1,1]);go(di3P,di3G,[gx(W3),gx(H3),220]);go(first[1].p,first[1].g,first[1].d);for(const op of rb[1])go(op.p,op.g,op.d);go(dc[1].p,dc[1].g,dc[1].d);go(fu3,fu3g,[gx(W3),gx(H3),4]);go(ft2P,ft2G,[Math.ceil(48*W2*H2/256),1,1]);go(di2P,di2G,[gx(W2),gx(H2),148]);go(first[2].p,first[2].g,first[2].d);for(const op of rb[2])go(op.p,op.g,op.d);go(dc[2].p,dc[2].g,dc[2].d);go(fu2,fu2g,[gx(W2),gx(H2),4]);go(ft1P,ft1G,[Math.ceil(32*W1*H1/256),1,1]);go(di1P,di1G,[gx(W1),gx(H1),100]);go(first[3].p,first[3].g,first[3].d);for(const op of rb[3])go(op.p,op.g,op.d);go(dc[3].p,dc[3].g,dc[3].d);go(fu1,fu1g,[gx(w),gx(h),4]);const out0=bind(outP,[final,flow0,outTex.createView()]);const out1=device.createBindGroup({layout:outP.getBindGroupLayout(1),entries:[{binding:0,resource:current.a.createView()},{binding:1,resource:current.b.createView()},{binding:2,resource:sampler}]});submit(outP,out0,[gx(outTex.width),gx(outTex.height),1],out1);}
  async function inspectFinal() {
    const read = device.createBuffer({ size: 8 * P * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(final, 0, read, 0, 8 * P * 4);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(read.getMappedRange().slice(0));
    let finite = 0, total = 0;
    for (const value of values) if (Number.isFinite(value)) { finite++; total += Math.abs(value); }
    read.unmap(); read.destroy();
    return { finite, total: total / Math.max(1, finite) };
  }
  return { prepPair, runT, inspectFinal, destroy(){all.forEach(b=>b.destroy());} };
}
