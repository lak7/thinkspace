import * as THREE from "three";

export const DISC_RADIUS = 2.2;
export const LAYER_GAP = 0.62;
export const TOKEN_RING_RADIUS = 3.5;

export function layerY(layer: number, nLayer: number): number {
  return (layer - (nLayer - 1) / 2) * LAYER_GAP;
}

/** A soft round sprite drawn in the fragment shader; shared by stars and tokens. */
export const POINT_VERTEX = /* glsl */ `
  attribute float aSize;
  attribute vec3 aColor;
  uniform float uPx;
  varying vec3 vColor;
  void main() {
    vColor = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = aSize * uPx / -mv.z;
  }
`;

export const POINT_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor * smoothstep(0.5, 0.05, d), 1.0);
  }
`;

/** Pixels per world unit at distance 1, for sizing points in world units. */
export function pixelScale(renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera): number {
  const h = renderer.getDrawingBufferSize(new THREE.Vector2()).y;
  return h / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
}
