import * as THREE from "three";
import type { Beam, Beams } from "./beams";
import type { Galaxy } from "./galaxy";
import type { TokenRing } from "./tokens";

export type Hit =
  | { kind: "star"; layer: number; neuron: number }
  | { kind: "beam"; beam: Beam }
  | { kind: "token"; index: number };

const STAR_RADIUS = 0.035;
const BEAM_RADIUS = 0.03;
const TOKEN_RADIUS = 0.12;

export class Picker {
  private readonly ray = new THREE.Raycaster();
  private readonly v = new THREE.Vector3();

  /**
   * `litLayers` limits star hits to layers currently showing data, so the
   * hover card never reports a value that is not on screen.
   */
  pick(
    ndc: THREE.Vector2,
    camera: THREE.Camera,
    galaxy: Galaxy,
    tokens: TokenRing,
    beams: Beams,
    litLayers: number,
  ): Hit | null {
    this.ray.setFromCamera(ndc, camera);

    this.ray.params.Points = { threshold: TOKEN_RADIUS };
    const t = this.ray.intersectObject(tokens.points, false)[0];
    if (t && t.index !== undefined) return { kind: "token", index: t.index };

    this.ray.params.Line = { threshold: BEAM_RADIUS };
    const b = this.ray.intersectObject(beams.lines, false)[0];
    if (b && b.index !== undefined) {
      const beam = beams.beamAt(b.index);
      if (beam) return { kind: "beam", beam };
    }

    // Stars: nearest to the ray among lit layers. A plain loop over 36,864
    // points is cheaper than a generic raycast and lets us filter by layer.
    const pos = galaxy.points.geometry.getAttribute("position").array as Float32Array;
    const n = litLayers * galaxy.dMlp;
    let best = -1;
    let bestD = STAR_RADIUS * STAR_RADIUS;
    for (let i = 0; i < n; i++) {
      this.v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      const d = this.ray.ray.distanceSqToPoint(this.v);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0) {
      return { kind: "star", layer: Math.floor(best / galaxy.dMlp), neuron: best % galaxy.dMlp };
    }
    return null;
  }
}
