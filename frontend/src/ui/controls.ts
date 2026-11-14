import { showToken } from "../protocol";

function el<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`Missing #${id}`);
  return e as T;
}

export interface RunParams {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  seed: number;
}

export interface ControlHandlers {
  run(params: RunParams): void;
  stop(): void;
  togglePlay(): void;
  stepToken(d: number): void;
  stepLayer(d: number): void;
  seek(token: number): void;
  speed(layersPerSecond: number): void;
  /** -1 means "follow the layer being lit". */
  layer(layer: number): void;
  heads(heads: Set<number>): void;
  beamThreshold(v: number): void;
  starThreshold(v: number): void;
  raw(on: boolean): void;
  reconnect(): void;
}

export type StatusKind = "ok" | "busy" | "bad" | "";

export class Controls {
  private readonly status = el<HTMLSpanElement>("status");
  private readonly fps = el<HTMLSpanElement>("fps");
  private readonly runBtn = el<HTMLButtonElement>("run");
  private readonly stopBtn = el<HTMLButtonElement>("stop");
  private readonly runInfo = el<HTMLSpanElement>("run-info");
  private readonly playBtn = el<HTMLButtonElement>("play");
  private readonly scrub = el<HTMLInputElement>("scrub");
  private readonly readout = el<HTMLSpanElement>("readout");
  private readonly strip = el<HTMLDivElement>("strip");
  private readonly banner = el<HTMLDivElement>("banner");
  private readonly layerChips: HTMLButtonElement[] = [];
  private readonly headChips: HTMLButtonElement[] = [];
  private readonly heads = new Set<number>();
  private stripCurrent = -1;

  constructor(nLayer: number, nHead: number, headCss: (h: number) => string, private readonly on: ControlHandlers) {
    const num = (id: string) => Number(el<HTMLInputElement>(id).value);
    const prompt = el<HTMLTextAreaElement>("prompt");
    const submit = () => {
      if (this.runBtn.disabled) return;
      on.run({
        prompt: prompt.value,
        max_new_tokens: num("max-new"),
        temperature: num("temperature"),
        top_k: num("top-k"),
        seed: num("seed"),
      });
    };
    this.runBtn.onclick = submit;
    prompt.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        submit();
      }
    });
    this.stopBtn.onclick = () => on.stop();

    this.playBtn.onclick = () => on.togglePlay();
    el("tok-prev").onclick = () => on.stepToken(-1);
    el("tok-next").onclick = () => on.stepToken(1);
    el("layer-prev").onclick = () => on.stepLayer(-1);
    el("layer-next").onclick = () => on.stepLayer(1);
    this.scrub.oninput = () => on.seek(Number(this.scrub.value));

    this.bindSlider("speed", "speed-out", (v) => `${v} layers/s`, on.speed);
    this.bindSlider("beam-thr", "beam-thr-out", (v) => v.toFixed(2), on.beamThreshold);
    this.bindSlider("star-thr", "star-thr-out", (v) => `${v.toFixed(2)}× scale`, on.starThreshold);
    const raw = el<HTMLInputElement>("raw");
    raw.onchange = () => on.raw(raw.checked);

    // Layer chips: Follow, then 1..nLayer.
    const layers = el("layers");
    for (let l = -1; l < nLayer; l++) {
      const b = document.createElement("button");
      b.className = "chip";
      b.textContent = l < 0 ? "Follow" : String(l + 1);
      b.title = l < 0 ? "Show attention for the layer currently lighting up" : `Show attention in layer ${l + 1}`;
      b.onclick = () => {
        this.setLayer(l);
        on.layer(l);
      };
      layers.append(b);
      this.layerChips.push(b);
    }
    this.setLayer(-1);

    const heads = el("heads");
    for (let h = 0; h < nHead; h++) {
      const b = document.createElement("button");
      b.className = "chip head on";
      b.textContent = String(h + 1);
      b.style.setProperty("--head", headCss(h));
      b.setAttribute("aria-pressed", "true");
      b.onclick = () => {
        if (this.heads.has(h)) this.heads.delete(h);
        else this.heads.add(h);
        this.paintHeads();
        on.heads(new Set(this.heads));
      };
      heads.append(b);
      this.headChips.push(b);
      this.heads.add(h);
    }
    const setAll = (all: boolean) => {
      this.heads.clear();
      if (all) for (let h = 0; h < nHead; h++) this.heads.add(h);
      this.paintHeads();
      on.heads(new Set(this.heads));
    };
    el("heads-all").onclick = () => setAll(true);
    el("heads-none").onclick = () => setAll(false);

    window.addEventListener("keydown", (e) => {
      const t = e.target as HTMLElement;
      if (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && (t as HTMLInputElement).type !== "range")) return;
      if (e.key === " ") on.togglePlay();
      else if (e.key === "ArrowRight") on.stepToken(1);
      else if (e.key === "ArrowLeft") on.stepToken(-1);
      else if (e.key === "ArrowUp") on.stepLayer(1);
      else if (e.key === "ArrowDown") on.stepLayer(-1);
      else return;
      e.preventDefault();
    });
  }

  private bindSlider(id: string, outId: string, fmt: (v: number) => string, cb: (v: number) => void): void {
    const input = el<HTMLInputElement>(id);
    const out = el<HTMLOutputElement>(outId);
    const update = () => {
      const v = Number(input.value);
      out.textContent = fmt(v);
      cb(v);
    };
    input.oninput = update;
    out.textContent = fmt(Number(input.value));
  }

  private setLayer(layer: number): void {
    this.layerChips.forEach((b, i) => b.classList.toggle("on", i - 1 === layer));
  }

  private paintHeads(): void {
    this.headChips.forEach((b, h) => {
      const on = this.heads.has(h);
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    });
  }

  setStatus(text: string, kind: StatusKind = ""): void {
    this.status.textContent = text;
    this.status.className = `pill ${kind}`;
  }

  setFps(fps: number): void {
    this.fps.textContent = `${fps.toFixed(0)} fps`;
  }

  setRunInfo(text: string): void {
    this.runInfo.textContent = text;
  }

  setRunning(running: boolean, connected: boolean): void {
    this.runBtn.disabled = running || !connected;
    this.stopBtn.disabled = !running;
  }

  setPlaying(playing: boolean): void {
    this.playBtn.textContent = playing ? "⏸" : "▶";
  }

  showBanner(text: string | null, action?: { label: string; run: () => void }): void {
    this.banner.replaceChildren();
    if (text === null) {
      this.banner.hidden = true;
      return;
    }
    const span = document.createElement("span");
    span.textContent = text;
    this.banner.append(span);
    if (action) {
      const b = document.createElement("button");
      b.textContent = action.label;
      b.onclick = action.run;
      this.banner.append(b);
    }
    this.banner.hidden = false;
  }

  clearTokens(): void {
    this.strip.replaceChildren();
    this.stripCurrent = -1;
    this.scrub.max = "0";
    this.scrub.value = "0";
  }

  addToken(index: number, token: string, isPrompt: boolean): void {
    const s = document.createElement("span");
    s.textContent = token.includes("\n") ? token.replace(/\n/g, "↵\n") : token;
    s.title = `#${index + 1} ${showToken(token)}`;
    s.className = (isPrompt ? "" : "gen ") + "future";
    s.onclick = () => this.on.seek(index);
    this.strip.append(s);
    this.scrub.max = String(index);
  }

  /** Update the readout, scrubber and token strip for the cursor. */
  setCursor(token: number, count: number, litLayers: number, nLayer: number): void {
    if (count === 0) {
      this.readout.textContent = "no run yet";
      return;
    }
    this.readout.textContent = `token ${token + 1}/${count} · layer ${litLayers}/${nLayer}`;
    this.scrub.value = String(token);
    if (token === this.stripCurrent) return;
    const spans = this.strip.children;
    const lo = Math.min(token, this.stripCurrent < 0 ? 0 : this.stripCurrent);
    const hi = Math.min(spans.length - 1, Math.max(token, this.stripCurrent));
    for (let i = lo; i <= hi; i++) {
      spans[i].classList.toggle("current", i === token);
      spans[i].classList.toggle("future", i > token);
    }
    this.stripCurrent = token;
    (spans[token] as HTMLElement | undefined)?.scrollIntoView({ block: "nearest" });
  }
}
