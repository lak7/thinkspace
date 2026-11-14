import type { Frame } from "./protocol";

/**
 * The recorded run and the playback cursor.
 *
 * The cursor is (token, layerF). layerF runs from 0 (nothing lit) to nLayer
 * (every layer lit), then holds for HOLD layer-units before moving to the
 * next token. Playback only replays recorded frames; it never creates values.
 */
export class Player {
  static readonly HOLD = 5;

  frames: Frame[] = [];
  token = 0;
  layerF = 0;
  playing = false;
  /** Layers per second. */
  speed = 8;
  /** True while the server is still sending frames. */
  streaming = false;

  private maxFromSecond = 0;
  private maxFirst = 0;

  constructor(readonly nLayer: number) {}

  reset(): void {
    this.frames = [];
    this.token = 0;
    this.layerF = 0;
    this.playing = false;
    this.maxFromSecond = 0;
    this.maxFirst = 0;
  }

  push(frame: Frame): void {
    this.frames.push(frame);
    let m = 0;
    for (let l = 1; l < frame.h.resid.length; l++) m = Math.max(m, frame.h.resid[l]);
    if (frame.h.pos === 0) this.maxFirst = m;
    else this.maxFromSecond = Math.max(this.maxFromSecond, m);
  }

  get frame(): Frame | null {
    return this.frames[this.token] ?? null;
  }

  /**
   * Largest post-block residual norm in the run. GPT-2's first position has a
   * far larger norm than every other one, so it is left out of the scale
   * whenever there is a second token; otherwise all rings but one look dark.
   */
  get residMax(): number {
    return this.maxFromSecond || this.maxFirst || 1;
  }

  /** Advance playback by dt seconds. Returns true if the cursor moved. */
  tick(dt: number): boolean {
    if (!this.playing || this.frames.length === 0) return false;
    const end = this.nLayer + Player.HOLD;
    const before = this.layerF;
    this.layerF = Math.min(this.layerF + this.speed * dt, end);
    if (this.layerF >= end) {
      if (this.token + 1 < this.frames.length) {
        this.token++;
        this.layerF = 0;
        return true;
      }
      if (!this.streaming) this.playing = false;
    }
    return this.layerF !== before;
  }

  /** Jump to a token with all layers lit. */
  seekToken(i: number): void {
    if (this.frames.length === 0) return;
    this.token = Math.min(Math.max(i, 0), this.frames.length - 1);
    this.layerF = this.nLayer;
  }

  stepLayer(d: number): void {
    if (this.frames.length === 0) return;
    let l = Math.min(Math.floor(this.layerF), this.nLayer) + d;
    if (l > this.nLayer) {
      if (this.token + 1 >= this.frames.length) l = this.nLayer;
      else {
        this.token++;
        l = 1;
      }
    } else if (l < 0) {
      if (this.token === 0) l = 0;
      else {
        this.token--;
        l = this.nLayer;
      }
    }
    this.layerF = l;
  }
}
