import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { showToken, type Frame, type Manifest } from "./protocol";
import { Player } from "./run";
import { Beams, headColor } from "./scene/beams";
import { Galaxy, type LayoutKind } from "./scene/galaxy";
import { pixelScale } from "./scene/geometry";
import { Picker, type Hit } from "./scene/picking";
import { Post } from "./scene/post";
import { TokenRing } from "./scene/tokens";
import { Socket } from "./socket";
import { showCandidates } from "./ui/candidates";
import { Controls } from "./ui/controls";
import { showHover } from "./ui/hover";

const LABELLED_SOURCES = 5;
const MORPH_SECONDS = 0.9;
const LAYOUT_KEY = "galaxy.layout";

async function fetchLayout(by: LayoutKind): Promise<Float32Array> {
  const l = await fetch(`/api/layout?by=${by}`);
  if (!l.ok) throw new Error((await l.json().catch(() => null))?.detail ?? `Server returned ${l.status}`);
  return new Float32Array(await l.arrayBuffer());
}

async function loadLayout(): Promise<{ manifest: Manifest; layout: Float32Array }> {
  const m = await fetch("/api/manifest");
  if (!m.ok) throw new Error((await m.json().catch(() => null))?.detail ?? `Server returned ${m.status}`);
  return { manifest: (await m.json()) as Manifest, layout: await fetchLayout("writes") };
}

async function boot(): Promise<void> {
  const canvas = document.getElementById("gl") as HTMLCanvasElement;
  const status = document.getElementById("status")!;
  let loaded: { manifest: Manifest; layout: Float32Array };
  try {
    status.textContent = "Loading star map…";
    loaded = await loadLayout();
  } catch (e) {
    status.textContent = "Not ready";
    status.className = "pill bad";
    const banner = document.getElementById("banner")!;
    banner.textContent = `Could not load the star map: ${(e as Error).message}`;
    banner.hidden = false;
    return;
  }
  const { manifest, layout } = loaded;
  const nLayer = manifest.n_layer;
  const nHead = manifest.n_head;

  // ---- Scene -------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x03040a);
  const labelRenderer = new CSS2DRenderer({ element: document.getElementById("labels") as HTMLElement });

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  camera.position.set(7, 3.6, 13.5);
  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = true;
  orbit.target.set(0, -1.5, 0); // leaves room for the bottom panel
  orbit.minDistance = 3;
  orbit.maxDistance = 40;

  const galaxy = new Galaxy(layout, manifest);
  const tokens = new TokenRing(nLayer);
  const beams = new Beams(nLayer, nHead);
  scene.add(galaxy.points, ...galaxy.rings, beams.lines, tokens.group);
  const post = new Post(renderer, scene, camera);
  const picker = new Picker();
  const player = new Player(nLayer);

  const view = {
    layer: -1, // -1 follows the layer being lit
    heads: new Set(Array.from({ length: nHead }, (_, h) => h)),
    beamThreshold: 0.05,
    starThreshold: 0,
    raw: false,
    layout: "writes" as LayoutKind,
  };
  const available = new Set<LayoutKind>(["writes"]);
  let morphT = 1; // 1 = not morphing

  let dirty = true;
  let running = false;
  let runStart = 0;
  let shown: Frame | null = null;
  let beamKey = "";
  const pointer = { ndc: new THREE.Vector2(), x: 0, y: 0, inside: false, moved: false };

  function resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    labelRenderer.setSize(w, h);
    post.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const px = pixelScale(renderer, camera);
    galaxy.setPixelScale(px);
    tokens.setPixelScale(px);
    dirty = true;
  }
  window.addEventListener("resize", resize);

  // ---- Controls ----------------------------------------------------------
  const socket = new Socket({
    onOpen() {
      ui.showBanner(null);
      ui.setStatus(`Ready · ${manifest.model} on your machine`, "ok");
      ui.setRunning(false, true);
    },
    onClose() {
      const wasRunning = running;
      running = false;
      player.streaming = false;
      ui.setRunning(false, false);
      ui.setStatus("Disconnected", "bad");
      ui.showBanner(
        wasRunning
          ? "The connection dropped mid-run. Frames received so far are still here."
          : "Not connected to the local server. Retrying every few seconds.",
        { label: "Reconnect now", run: () => socket.connect() },
      );
    },
    onFrame(frame) {
      if (player.frames.length === 0) {
        ui.setRunInfo(`first token after ${(performance.now() - runStart).toFixed(0)} ms`);
      }
      player.push(frame);
      tokens.add(frame.h.token, frame.h.is_prompt);
      ui.addToken(frame.h.pos, frame.h.token, frame.h.is_prompt);
      // Start playback on the last prompt token: the one that predicts the
      // first generated token. Earlier prompt tokens stay scrubbable.
      if (frame.h.is_prompt) {
        player.token = frame.h.pos;
        player.layerF = 0;
      }
      dirty = true;
    },
    onMessage(msg) {
      if (msg.type === "started") {
        player.reset();
        player.streaming = true;
        player.playing = true;
        tokens.clear(msg.expected_tokens);
        ui.clearTokens();
        ui.setPlaying(true);
        shown = null;
        beamKey = "";
      } else {
        running = false;
        player.streaming = false;
        ui.setRunning(false, socket.open);
        if (msg.type === "done") {
          ui.setStatus(`Ready · ${manifest.model} on your machine`, "ok");
          ui.setRunInfo(`${msg.frames} tokens${msg.stopped ? " (stopped)" : ""}`);
        } else {
          ui.setStatus("Run failed", "bad");
          ui.setRunInfo("");
          ui.showBanner(msg.message, { label: "Dismiss", run: () => ui.showBanner(null) });
        }
      }
      dirty = true;
    },
  });

  const ui = new Controls(nLayer, nHead, (h) => `#${headColor(h, nHead).getHexString()}`, {
    run(params) {
      if (!params.prompt) {
        ui.showBanner("Type a prompt first.", { label: "OK", run: () => ui.showBanner(null) });
        return;
      }
      ui.showBanner(null);
      runStart = performance.now();
      if (!socket.send({ type: "run", ...params })) return;
      running = true;
      ui.setRunning(true, true);
      ui.setStatus("Running the model…", "busy");
      ui.setRunInfo("");
    },
    stop: () => void socket.send({ type: "stop" }),
    togglePlay() {
      if (player.frames.length === 0) return;
      const atEnd = player.token === player.frames.length - 1 && player.layerF >= nLayer;
      if (!player.playing && atEnd && !player.streaming) {
        player.token = 0;
        player.layerF = 0;
      }
      player.playing = !player.playing;
      dirty = true;
    },
    stepToken(d) {
      player.playing = false;
      player.seekToken(player.token + d);
      dirty = true;
    },
    stepLayer(d) {
      player.playing = false;
      player.stepLayer(d);
      dirty = true;
    },
    seek(token) {
      player.playing = false;
      player.seekToken(token);
      dirty = true;
    },
    speed: (v) => void (player.speed = v),
    layer(l) {
      view.layer = l;
      dirty = true;
    },
    heads(h) {
      view.heads = h;
      dirty = true;
    },
    beamThreshold(v) {
      view.beamThreshold = v;
      dirty = true;
    },
    starThreshold(v) {
      view.starThreshold = v;
      galaxy.setThreshold(v);
      dirty = true;
    },
    raw(on) {
      view.raw = on;
      post.raw = on;
      dirty = true;
    },
    layout(kind) {
      if (kind === view.layout || !available.has(kind)) return;
      view.layout = kind;
      try {
        localStorage.setItem(LAYOUT_KEY, kind);
      } catch {
        /* storage blocked: the choice just isn't remembered */
      }
      galaxy.beginMorph(kind);
      // Raw mode shows only data-driven pixels, so jump instead of animating.
      morphT = view.raw ? 1 : 0;
      galaxy.morph(morphT);
      ui.setLayout(kind, available);
      dirty = true;
    },
    reconnect: () => socket.connect(),
  });
  ui.setLayout(view.layout, available);

  // The "reads" map is optional (older precomputes lack it), so load it after first paint.
  void fetchLayout("reads")
    .then((reads) => {
      galaxy.addLayout("reads", reads);
      available.add("reads");
      let saved: string | null = null;
      try {
        saved = localStorage.getItem(LAYOUT_KEY);
      } catch {
        /* ignore */
      }
      if (saved === "reads") {
        view.layout = "reads";
        galaxy.beginMorph("reads");
        galaxy.morph(1);
        dirty = true;
      }
      ui.setLayout(view.layout, available);
    })
    .catch(() => ui.setLayout(view.layout, available));

  // ---- Applying the cursor to the scene ----------------------------------
  /** Layer cursor as drawn: raw mode snaps to whole layers (no fade-in). */
  function drawnCursor(): number {
    const c = Math.min(player.layerF, nLayer);
    return view.raw ? Math.floor(c) : c;
  }

  function apply(): void {
    const frame = player.frame;
    if (frame !== shown) {
      galaxy.setActivations(frame ? frame.mlp : null);
      tokens.setCurrent(frame ? frame.h.pos : -1);
      shown = frame;
    }
    const cursor = frame ? drawnCursor() : 0;
    const lit = Math.ceil(cursor); // layers showing data
    galaxy.setCursor(cursor);

    // Attention is drawn for one layer: the newest lit layer, or the pinned one.
    const beamLayer = view.layer < 0 ? lit - 1 : lit > view.layer ? view.layer : -1;
    const key = `${frame?.h.pos}|${beamLayer}|${[...view.heads].join(",")}|${view.beamThreshold}`;
    if (key !== beamKey) {
      beamKey = key;
      beams.update(frame, beamLayer, view.heads, view.beamThreshold, tokens);
    }
    const labelled = new Set<number>();
    if (frame) {
      labelled.add(frame.h.pos);
      [...beams.beams]
        .sort((a, b) => b.weight - a.weight)
        .slice(0, LABELLED_SOURCES * 2)
        .forEach((b) => labelled.size <= LABELLED_SOURCES && labelled.add(b.to));
    }
    tokens.showLabels(labelled);

    galaxy.setResidual(frame ? frame.h.resid : null, player.residMax, cursor, view.layer);
    showCandidates(frame, player.frames[player.token + 1] ?? null, cursor >= nLayer, player.residMax);
    ui.setCursor(player.token, player.frames.length, lit, nLayer);
    ui.setPlaying(player.playing);
  }

  // ---- Hover -------------------------------------------------------------
  function describe(hit: Hit): void {
    const frame = player.frame;
    const tokText = (i: number) => `#${i + 1} “${showToken(player.frames[i]?.h.token ?? "")}”`;
    if (hit.kind === "star") {
      const rows = [
        { label: "Neuron", value: `${hit.neuron} of ${galaxy.dMlp}` },
        { label: "Placed by", value: view.layout === "reads" ? "input weights" : "output weights" },
      ];
      if (frame) {
        const act = galaxy.activation(hit.layer, hit.neuron);
        const scale = manifest.scales[hit.layer];
        rows.push(
          { label: "Activation", value: act.toFixed(4) },
          { label: "Layer scale (p99)", value: scale.toFixed(3) },
          { label: "Brightness", value: `${(Math.abs(act) / scale).toFixed(2)}×` },
          { label: "Token", value: tokText(frame.h.pos) },
        );
      } else {
        rows.push({ label: "Activation", value: "no run yet" });
      }
      showHover(`MLP neuron · layer ${hit.layer + 1}`, rows, pointer.x, pointer.y);
    } else if (hit.kind === "beam") {
      const b = hit.beam;
      showHover(
        `Attention · layer ${b.layer + 1}, head ${b.head + 1}`,
        [
          { label: "Weight", value: b.weight.toFixed(4) },
          { label: "From", value: tokText(b.from) },
          { label: b.to === b.from ? "To (itself)" : "To", value: tokText(b.to) },
        ],
        pointer.x,
        pointer.y,
      );
    } else {
      const h = player.frames[hit.index]?.h;
      if (!h) return showHover(null);
      showHover(
        `Token #${hit.index + 1}`,
        [
          { label: "Text", value: `“${showToken(h.token)}”` },
          { label: "Vocabulary id", value: String(h.token_id) },
          { label: "Source", value: h.is_prompt ? "your prompt" : "generated" },
        ],
        pointer.x,
        pointer.y,
      );
    }
  }

  canvas.addEventListener("pointermove", (e) => {
    pointer.ndc.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    pointer.inside = true;
    pointer.moved = true;
  });
  canvas.addEventListener("pointerleave", () => {
    pointer.inside = false;
    showHover(null);
  });
  canvas.addEventListener("click", () => {
    if (!pointer.inside) return;
    const hit = picker.pick(pointer.ndc, camera, galaxy, tokens, beams, 0);
    if (hit?.kind === "token") {
      player.playing = false;
      player.seekToken(hit.index);
      dirty = true;
    }
  });

  // ---- Frame loop --------------------------------------------------------
  let last = performance.now();
  let fpsFrames = 0;
  let fpsSince = last;
  function tick(now: number): void {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    let moved = false;
    if (morphT < 1) {
      morphT = Math.min(morphT + dt / MORPH_SECONDS, 1);
      galaxy.morph(morphT);
      moved = true;
    }
    const changed = player.tick(dt) || dirty || moved;
    if (changed) {
      apply();
      dirty = false;
    }
    orbit.update();
    if (pointer.inside && (pointer.moved || changed)) {
      pointer.moved = false;
      const litLayers = player.frame ? Math.ceil(drawnCursor()) : nLayer;
      const hit = picker.pick(pointer.ndc, camera, galaxy, tokens, beams, litLayers);
      if (hit) describe(hit);
      else showHover(null);
    }
    post.render();
    labelRenderer.render(scene, camera);

    fpsFrames++;
    if (now - fpsSince >= 500) {
      ui.setFps((fpsFrames * 1000) / (now - fpsSince));
      fpsFrames = 0;
      fpsSince = now;
    }
    requestAnimationFrame(tick);
  }

  resize();
  ui.setStatus("Connecting…");
  socket.connect();
  requestAnimationFrame(tick);

  // Handy for checking displayed values against the backend from the console.
  Object.assign(window, { galaxyDebug: { player, galaxy, beams, manifest } });
}

void boot();
