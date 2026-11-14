import * as THREE from "three";
import type { Manifest } from "../protocol";
import { DISC_RADIUS, layerY } from "./geometry";

// Every star is one MLP neuron. Position comes from the precomputed UMAP of
// the neuron's output weights; brightness is |activation| / layer scale for
// the current token; colour is the activation's sign.
const VERT = /* glsl */ `
  attribute float aLayer;
  attribute float aScale;
  attribute float aAct;
  uniform float uCursor;     // layers below this are lit (fractional = fade-in)
  uniform float uThreshold;  // hide stars whose scaled activation is below this
  uniform float uSize;
  uniform float uPx;
  varying float vV;
  varying float vSign;
  void main() {
    float v = abs(aAct) / aScale;
    if (v < uThreshold) v = 0.0;
    v *= clamp(uCursor - aLayer, 0.0, 1.0);
    vV = v;
    vSign = aAct >= 0.0 ? 1.0 : -1.0;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * (0.6 + 1.5 * sqrt(min(v, 4.0))) * uPx / -mv.z;
  }
`;

const FRAG = /* glsl */ `
  varying float vV;
  varying float vSign;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    vec3 unlit = vec3(0.032, 0.043, 0.08);
    vec3 pos = vec3(1.0, 0.74, 0.36);
    vec3 neg = vec3(0.30, 0.52, 1.0);
    // Linear in the scaled activation, capped so one extreme neuron cannot wash out a disc.
    vec3 col = unlit + (vSign > 0.0 ? pos : neg) * min(vV, 2.5) * 0.6;
    gl_FragColor = vec4(col * smoothstep(0.5, 0.05, d), 1.0);
  }
`;

export class Galaxy {
  readonly points: THREE.Points;
  readonly rings: THREE.LineLoop[] = [];
  readonly nLayer: number;
  readonly dMlp: number;
  private readonly act: THREE.BufferAttribute;
  private readonly material: THREE.ShaderMaterial;

  constructor(layout: Float32Array, manifest: Manifest) {
    this.nLayer = manifest.n_layer;
    this.dMlp = manifest.d_mlp;
    const n = this.nLayer * this.dMlp;
    if (layout.length !== n * 2) throw new Error("layout.bin does not match the manifest");

    const position = new Float32Array(n * 3);
    const layer = new Float32Array(n);
    const scale = new Float32Array(n);
    for (let l = 0; l < this.nLayer; l++) {
      const y = layerY(l, this.nLayer);
      for (let i = 0; i < this.dMlp; i++) {
        const k = l * this.dMlp + i;
        position[k * 3] = layout[k * 2] * DISC_RADIUS;
        position[k * 3 + 1] = y;
        position[k * 3 + 2] = layout[k * 2 + 1] * DISC_RADIUS;
        layer[k] = l;
        scale[k] = manifest.scales[l];
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(position, 3));
    geo.setAttribute("aLayer", new THREE.BufferAttribute(layer, 1));
    geo.setAttribute("aScale", new THREE.BufferAttribute(scale, 1));
    this.act = new THREE.BufferAttribute(new Float32Array(n), 1);
    this.act.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aAct", this.act);
    geo.computeBoundingSphere();

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uCursor: { value: 0 },
        uThreshold: { value: 0 },
        uSize: { value: 0.026 },
        uPx: { value: 1 },
      },
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;

    // One ring per layer; its brightness is that layer's residual-stream norm.
    const circle: THREE.Vector3[] = [];
    for (let i = 0; i < 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      circle.push(new THREE.Vector3(Math.cos(a) * DISC_RADIUS * 1.06, 0, Math.sin(a) * DISC_RADIUS * 1.06));
    }
    const ringGeo = new THREE.BufferGeometry().setFromPoints(circle);
    for (let l = 0; l < this.nLayer; l++) {
      const ring = new THREE.LineLoop(
        ringGeo,
        new THREE.LineBasicMaterial({
          color: 0x000000,
          blending: THREE.AdditiveBlending,
          transparent: true,
          depthTest: false,
        }),
      );
      ring.position.y = layerY(l, this.nLayer);
      this.rings.push(ring);
    }
  }

  /** Load one token's real activations (or clear with null). */
  setActivations(mlp: Float32Array | null): void {
    const arr = this.act.array as Float32Array;
    if (mlp) arr.set(mlp);
    else arr.fill(0);
    this.act.needsUpdate = true;
  }

  activation(layer: number, neuron: number): number {
    return (this.act.array as Float32Array)[layer * this.dMlp + neuron];
  }

  setCursor(layerCursor: number): void {
    this.material.uniforms.uCursor.value = layerCursor;
  }

  setThreshold(t: number): void {
    this.material.uniforms.uThreshold.value = t;
  }

  setPixelScale(px: number): void {
    this.material.uniforms.uPx.value = px;
  }

  /**
   * resid[l + 1] is the residual norm after block l. `max` is the largest
   * such norm in the run, so ring brightness is comparable across tokens.
   */
  setResidual(resid: number[] | null, max: number, layerCursor: number, selected: number): void {
    for (let l = 0; l < this.nLayer; l++) {
      const mat = this.rings[l].material as THREE.LineBasicMaterial;
      const lit = Math.min(Math.max(layerCursor - l, 0), 1);
      const v = resid ? Math.min(resid[l + 1] / max, 1) * lit : 0;
      const sel = l === selected ? 0.16 : 0;
      mat.color.setRGB(0.05 + 0.5 * v + sel, 0.07 + 0.75 * v + sel, 0.12 + 0.9 * v + sel);
    }
  }
}
