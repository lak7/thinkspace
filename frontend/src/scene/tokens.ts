import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { showToken } from "../protocol";
import { LAYER_GAP, POINT_FRAGMENT, POINT_VERTEX, TOKEN_RING_RADIUS, layerY } from "./geometry";

const CAPACITY = 1024;
const MAX_STEP = 0.15; // radians between neighbouring tokens
const MAX_SPAN = 5.4; // the ring never closes, so first and last stay distinct
const LABEL_ALL_LIMIT = 48;

const PROMPT_COLOR = new THREE.Color(0.55, 0.6, 0.72);
const GENERATED_COLOR = new THREE.Color(0.35, 0.85, 0.95);
const CURRENT_COLOR = new THREE.Color(1.6, 1.6, 1.6);

/** Tokens of the run, laid out on an arc around the base of the layer stack. */
export class TokenRing {
  readonly group = new THREE.Group();
  readonly points: THREE.Points;
  count = 0;
  private readonly y: number;
  private step = MAX_STEP;
  private span = 0;
  private readonly isPrompt: boolean[] = [];
  private readonly labels: CSS2DObject[] = [];
  private readonly pos: THREE.BufferAttribute;
  private readonly color: THREE.BufferAttribute;
  private readonly size: THREE.BufferAttribute;
  private readonly material: THREE.ShaderMaterial;
  private current = -1;

  constructor(nLayer: number) {
    this.y = layerY(0, nLayer) - LAYER_GAP * 1.6;
    const geo = new THREE.BufferGeometry();
    this.pos = new THREE.BufferAttribute(new Float32Array(CAPACITY * 3), 3);
    this.color = new THREE.BufferAttribute(new Float32Array(CAPACITY * 3), 3);
    this.size = new THREE.BufferAttribute(new Float32Array(CAPACITY), 1);
    geo.setAttribute("position", this.pos);
    geo.setAttribute("aColor", this.color);
    geo.setAttribute("aSize", this.size);
    geo.setDrawRange(0, 0);
    this.material = new THREE.ShaderMaterial({
      vertexShader: POINT_VERTEX,
      fragmentShader: POINT_FRAGMENT,
      uniforms: { uPx: { value: 1 } },
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    this.group.add(this.points);
  }

  setPixelScale(px: number): void {
    this.material.uniforms.uPx.value = px;
  }

  clear(expected: number): void {
    for (const l of this.labels) {
      this.group.remove(l);
      l.element.remove();
    }
    this.labels.length = 0;
    this.isPrompt.length = 0;
    this.count = 0;
    this.current = -1;
    const n = Math.max(expected, 2);
    this.step = Math.min(MAX_STEP, MAX_SPAN / (n - 1));
    this.span = this.step * (n - 1);
    this.points.geometry.setDrawRange(0, 0);
  }

  /** World position of token i. Token 0 sits on the left, reading clockwise. */
  position(i: number, out = new THREE.Vector3()): THREE.Vector3 {
    const a = Math.PI / 2 + this.span / 2 - i * this.step;
    return out.set(Math.cos(a) * TOKEN_RING_RADIUS, this.y, Math.sin(a) * TOKEN_RING_RADIUS);
  }

  add(token: string, isPrompt: boolean): void {
    if (this.count >= CAPACITY) return;
    const i = this.count++;
    const p = this.position(i);
    this.pos.setXYZ(i, p.x, p.y, p.z);
    this.isPrompt.push(isPrompt);
    const el = document.createElement("div");
    el.className = "tok-label" + (isPrompt ? "" : " gen");
    el.textContent = showToken(token);
    const label = new CSS2DObject(el);
    label.position.copy(p);
    label.center.set(0.5, 1.6);
    label.visible = false;
    this.labels.push(label);
    this.group.add(label);
    this.paint(i);
    this.pos.needsUpdate = true;
    this.points.geometry.setDrawRange(0, this.count);
  }

  private paint(i: number): void {
    const c = i === this.current ? CURRENT_COLOR : this.isPrompt[i] ? PROMPT_COLOR : GENERATED_COLOR;
    // Tokens after the cursor have not been "reached" yet in playback: dimmed.
    const k = i > this.current ? 0.3 : 1;
    this.color.setXYZ(i, c.r * k, c.g * k, c.b * k);
    this.size.setX(i, i === this.current ? 0.2 : 0.11);
    this.color.needsUpdate = true;
    this.size.needsUpdate = true;
  }

  setCurrent(i: number): void {
    const prev = this.current;
    this.current = i;
    const lo = Math.max(0, Math.min(prev, i));
    const hi = Math.min(this.count - 1, Math.max(prev, i));
    for (let t = lo; t <= hi; t++) this.paint(t);
  }

  /** Labels: all when the run is short, otherwise only the listed tokens. */
  showLabels(indices: Set<number>): void {
    const all = this.count <= LABEL_ALL_LIMIT;
    for (let i = 0; i < this.labels.length; i++) {
      const on = all || indices.has(i);
      this.labels[i].visible = on;
      this.labels[i].element.classList.toggle("current", i === this.current);
      this.labels[i].element.classList.toggle("future", i > this.current);
    }
  }
}
