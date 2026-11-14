import * as THREE from "three";
import type { Frame } from "../protocol";
import { layerY } from "./geometry";
import type { TokenRing } from "./tokens";

const SEGMENTS = 20;
const MAX_BEAMS = 1300;
const SELF_SPIKE = 0.55;

export interface Beam {
  layer: number;
  head: number;
  from: number; // the token doing the attending (the current token)
  to: number; // the token being attended to
  weight: number;
}

export function headColor(head: number, nHead: number, out = new THREE.Color()): THREE.Color {
  return out.setHSL(head / nHead, 0.85, 0.6);
}

/**
 * One curve per (head, earlier token) for the chosen layer. A curve's colour
 * intensity equals the attention weight, so with additive blending a weight
 * of 0.5 is half as bright as a weight of 1. Attention to the token itself
 * is drawn as a short vertical spike above it.
 */
export class Beams {
  readonly lines: THREE.LineSegments;
  beams: Beam[] = [];
  private readonly pos: THREE.BufferAttribute;
  private readonly col: THREE.BufferAttribute;

  constructor(private readonly nLayer: number, private readonly nHead: number) {
    const geo = new THREE.BufferGeometry();
    const verts = MAX_BEAMS * SEGMENTS * 2;
    this.pos = new THREE.BufferAttribute(new Float32Array(verts * 3), 3);
    this.col = new THREE.BufferAttribute(new Float32Array(verts * 3), 3);
    this.pos.setUsage(THREE.DynamicDrawUsage);
    this.col.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", this.pos);
    geo.setAttribute("color", this.col);
    geo.setDrawRange(0, 0);
    this.lines = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthTest: false,
      }),
    );
    this.lines.frustumCulled = false;
  }

  /** Beam index for a raycast hit on `lines` (hit.index is a vertex index). */
  beamAt(vertexIndex: number): Beam | undefined {
    return this.beams[Math.floor(vertexIndex / (SEGMENTS * 2))];
  }

  update(frame: Frame | null, layer: number, heads: Set<number>, minWeight: number, ring: TokenRing): void {
    this.beams = [];
    if (!frame || layer < 0) {
      this.lines.geometry.setDrawRange(0, 0);
      return;
    }
    const seq = frame.h.seq_len;
    const cur = frame.h.pos;
    const base = layer * this.nHead * seq;
    const p0 = new THREE.Vector3();
    const p2 = ring.position(cur);
    const ctrl = new THREE.Vector3();
    const hub = new THREE.Vector3(0, layerY(layer, this.nLayer), 0);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const color = new THREE.Color();
    const P = this.pos.array as Float32Array;
    const C = this.col.array as Float32Array;
    let v = 0;

    for (const head of heads) {
      headColor(head, this.nHead, color);
      for (let j = 0; j < seq && this.beams.length < MAX_BEAMS; j++) {
        const w = frame.attn[base + head * seq + j];
        if (!(w >= minWeight)) continue;
        this.beams.push({ layer, head, from: cur, to: j, weight: w });
        if (j === cur) {
          p0.copy(p2);
          ctrl.copy(p2).setY(p2.y + SELF_SPIKE / 2);
        } else {
          ring.position(j, p0);
          // Bend the curve towards the centre of the layer doing the attending.
          ctrl.addVectors(p0, p2).multiplyScalar(0.5).lerp(hub, 0.8);
        }
        const end = j === cur ? b.copy(p2).setY(p2.y + SELF_SPIKE).clone() : p2;
        a.copy(p0);
        for (let s = 1; s <= SEGMENTS; s++) {
          const t = s / SEGMENTS;
          const u = 1 - t;
          b.set(
            u * u * p0.x + 2 * u * t * ctrl.x + t * t * end.x,
            u * u * p0.y + 2 * u * t * ctrl.y + t * t * end.y,
            u * u * p0.z + 2 * u * t * ctrl.z + t * t * end.z,
          );
          P[v * 3] = a.x; P[v * 3 + 1] = a.y; P[v * 3 + 2] = a.z;
          C[v * 3] = color.r * w; C[v * 3 + 1] = color.g * w; C[v * 3 + 2] = color.b * w;
          v++;
          P[v * 3] = b.x; P[v * 3 + 1] = b.y; P[v * 3 + 2] = b.z;
          C[v * 3] = color.r * w; C[v * 3 + 1] = color.g * w; C[v * 3 + 2] = color.b * w;
          v++;
          a.copy(b);
        }
      }
    }
    this.pos.needsUpdate = true;
    this.col.needsUpdate = true;
    this.lines.geometry.setDrawRange(0, v);
  }
}
