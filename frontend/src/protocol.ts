// Decoder for the binary frame format defined in backend/galaxy/protocol.py.

export interface Candidate {
  id: number;
  token: string;
  p: number;
}

export interface FrameHeader {
  v: number;
  pos: number;
  token_id: number;
  token: string;
  is_prompt: boolean;
  seq_len: number;
  n_layer: number;
  n_head: number;
  d_mlp: number;
  resid: number[];
  top: Candidate[];
}

export interface Frame {
  h: FrameHeader;
  /** n_layer * d_mlp post-GELU activations, layer-major. */
  mlp: Float32Array;
  /** n_layer * d_mlp pre-GELU scores, layer-major. Can be strongly negative. */
  pre: Float32Array;
  /** n_layer * n_head * seq_len attention weights from this token. */
  attn: Float32Array;
}

export interface Manifest {
  model: string;
  n_layer: number;
  n_head: number;
  d_model: number;
  d_mlp: number;
  scales: number[];
  scale_rule: string;
  /** p99 of |pre-GELU score| per layer; absent in older precomputes. */
  pre_scales?: number[];
  layout_rule: string;
  /** Present when precompute wrote both star maps. */
  layouts?: Record<string, { file: string; rule: string }>;
}

const EXP = new Float64Array(32);
for (let e = 0; e < 32; e++) EXP[e] = Math.pow(2, e - 15);
const SUBNORMAL = Math.pow(2, -14) / 1024;

function halfToFloat32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i];
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    let v: number;
    if (e === 0) v = f * SUBNORMAL;
    else if (e === 31) v = f ? NaN : Infinity;
    else v = EXP[e] * (1 + f / 1024);
    out[i] = h & 0x8000 ? -v : v;
  }
  return out;
}

export function decodeFrame(buf: ArrayBuffer): Frame {
  const view = new DataView(buf);
  const hlen = view.getUint32(0, true);
  const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen))) as FrameHeader;
  let off = 4 + hlen;
  off += (4 - (off % 4)) % 4;
  const nMlp = h.n_layer * h.d_mlp;
  const nAttn = h.n_layer * h.n_head * h.seq_len;
  if (h.v !== 2) throw new Error(`Frame version ${h.v} is not supported; rebuild the UI`);
  if (off + (2 * nMlp + nAttn) * 2 !== buf.byteLength) throw new Error("Malformed frame");
  const mlp = halfToFloat32(new Uint16Array(buf, off, nMlp));
  const pre = halfToFloat32(new Uint16Array(buf, off + nMlp * 2, nMlp));
  const attn = halfToFloat32(new Uint16Array(buf, off + nMlp * 4, nAttn));
  return { h, mlp, pre, attn };
}

/** Printable form of a token: makes leading spaces and newlines visible. */
export function showToken(t: string): string {
  if (t === "") return "∅";
  return t.replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/ /g, "·");
}
