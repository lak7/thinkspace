# PRD: LLM Galaxy (working title)

**Status:** Draft v0.1
**Date:** 2026-11-10
**Scope of this document:** v0, with a short roadmap for what follows

---

## 1. Summary

LLM Galaxy is a local app that shows what a language model is actually doing while it answers your prompt. Every neuron is a star, every layer is a galaxy, and every attention head draws beams between tokens. You type a prompt, the model runs on your machine, and the galaxy lights up with the real values from that forward pass.

The one rule that defines the product: **nothing on screen is simulated.** Every brightness, colour and beam is a number read out of the running model.

## 2. Problem

- People use LLMs daily but have never seen inside one. Existing explanations are static diagrams or fake "neural network" animations.
- Existing real visualisers (bbycroft's LLM Visualization, Transformer Explainer) are fixed walkthroughs of one model. You cannot point them at a model on your own machine and watch your own prompt.
- Interpretability tools (TransformerLens, Neuronpedia) are real but built for researchers, not for looking at.

## 3. Goals and non-goals

### Goals for v0

1. Run GPT-2 small locally and capture its real activations for any prompt.
2. Render all 36,864 MLP neurons and 144 attention heads as an interactive 3D galaxy at 60 fps.
3. Let the user step through generation token by token and layer by layer.
4. Make every visual element traceable to a tensor value (hover shows the number).
5. Look good enough that a screen recording is worth sharing.

### Non-goals for v0

- Explaining what a neuron "means" (needs sparse autoencoders; see roadmap).
- Models other than GPT-2 small.
- llama.cpp, GGUF, MLX or Ollama support.
- Hosting, accounts, or running the model in the browser.
- Editing or steering activations.

## 4. Target users

| User | What they want |
|---|---|
| Curious developer | To see what happens inside a model on their own prompt |
| Student or educator | A truthful visual to teach attention and MLP layers |
| Content creator | A striking, real visual to record |
| Interpretability hobbyist | A fast way to spot interesting neurons and heads before deeper analysis |

## 5. The "no simulation" contract

This is a hard requirement, not a style preference.

**Allowed**
- Any visual property driven by a captured tensor value.
- Star positions computed from the model's real weights.
- Slowed or stepped playback of recorded real values (a real forward pass takes milliseconds, so it must be replayed slower to be visible).
- Bloom, camera motion and fade-out easing as pure rendering effects.

**Not allowed**
- Random, procedural or looping animation that implies model activity.
- Interpolated values presented as activations. Fades between two real frames must not be hoverable as data.
- Particles or pulses that do not correspond to a captured value.

**Enforcement**
- Hovering any star or beam shows the raw value, layer, index and token position.
- A "raw mode" toggle turns off bloom and easing so only data-driven pixels remain.
- An automated test compares the values the frontend received against a fresh forward pass for the same prompt and seed.

## 6. What is visualised

Model facts for GPT-2 small: 12 layers, 12 heads per layer, hidden size 768, 3,072 MLP neurons per layer, 124M parameters, 50,257-token vocabulary.

| Visual element | Real source | Shape per generated token |
|---|---|---|
| Star (one per MLP neuron) | Post-GELU activation in each block's MLP | 12 × 3,072 |
| Star position | UMAP of each neuron's output weight vector (its row in the MLP output projection), or, toggleable, of its input weight vector (its column in the MLP input projection, with the LayerNorm gain folded in) | Precomputed once |
| Star brightness | Absolute activation, scaled per layer; or, toggleable, absolute pre-GELU score, scaled per layer | — |
| Star colour | Sign of the shown value. Post-GELU is never below about −0.17, so only the pre-GELU score shows strong suppression | — |
| Token stars | The prompt and generated tokens, laid out in a row | — |
| Attention beam | Attention weight from the current token to each earlier token, per head | 12 × 12 × sequence length |
| Layer pulse | L2 norm of the residual stream after each block | 13 values |
| Next-token panel | Top 10 output probabilities | 10 values |

**Brightness scaling:** a fixed scale per layer, set to the 99th percentile of activations over a small calibration text set, computed during the precompute step. This keeps brightness comparable across tokens and prompts. A per-frame auto-scale would be simpler but would make a quiet token look as busy as a loud one.

**Spatial layout:** each layer is a disc of 3,072 stars. The 12 discs are stacked along one axis, so a forward pass reads as light travelling from the first galaxy to the last. Within a disc, neurons that write similar directions into the residual stream sit near each other.

## 7. User experience

### Main flow

1. User starts the app and opens it in the browser. The galaxy is visible but dark.
2. User types a prompt and presses Run.
3. Prompt tokens appear as a row of token stars.
4. For each generated token, layers light up in order from 1 to 12. Attention beams for the selected layer connect the current token to earlier ones.
5. The next-token panel shows the top candidates, and the chosen token joins the row.
6. User can pause, scrub back to any token and layer, and orbit the camera freely.

### Controls

- **Prompt box** with Run and Stop.
- **Playback:** play, pause, step token, step layer, speed slider.
- **Timeline scrubber** over tokens and layers of the current run.
- **Layer and head selector** to filter which attention beams are drawn.
- **Threshold slider** to hide stars and beams below a value.
- **Hover card** on any star or beam with its raw value and coordinates.
- **Raw mode** toggle (see section 5).
- **Sampling settings:** temperature, top-k, seed, max new tokens.

### Empty and error states

- Model still loading: progress indicator with the download size.
- Backend disconnected: banner with a reconnect button; the last run stays scrubbable.
- Prompt longer than the 1,024-token context: rejected with the token count shown.

## 8. Architecture

```
Browser (three.js)  <── WebSocket (binary frames) ──>  Python backend (FastAPI + PyTorch)
      │                                                        │
  layout.bin (star positions, loaded once)            GPT-2 small + forward hooks
```

### Backend

- **Stack:** Python, PyTorch, Hugging Face `transformers`, FastAPI with WebSocket.
- **Device:** Apple MPS when available, otherwise CPU.
- **Capture:**
  - MLP activations via forward hooks on each block's activation module.
  - Attention weights via `output_attentions=True`, with the model loaded using the eager attention implementation (the optimised path does not return weights).
  - Residual norms from `output_hidden_states=True`.
- **Generation:** a manual token-by-token loop with the KV cache, so each step returns values for the new token only.
- **Run recording:** every run is kept in memory so the client can scrub without re-running the model.

### Precompute step (run once)

1. For each layer, take the 3,072 neuron output weight vectors (768 dimensions each).
2. Reduce to 2D with UMAP using a fixed seed.
3. Run the calibration text set and record each layer's 99th-percentile activation.
4. Write positions and scales to `layout.bin` plus a small JSON manifest.

### Wire protocol

One binary message per generated token:

| Section | Content | Size |
|---|---|---|
| Header (JSON) | Token id, token text, position, top-10 candidates, residual norms | under 1 KB |
| MLP block | 36,864 values as float16 | about 74 KB |
| Attention block | 144 × sequence-length values as float16 | about 29 KB at 100 tokens |

Roughly 100 KB per token, which is trivial over localhost.

### Frontend

- **Stack:** TypeScript, Vite, three.js.
- **Stars:** a single `Points` object with a custom shader. Positions are static; a per-star activation attribute is updated each frame from the received buffer.
- **Beams:** line segments with opacity from attention weight, drawn only for the selected layer and heads.
- **Post-processing:** bloom, disabled in raw mode.
- **Picking:** GPU colour picking or a spatial index for hover.

## 9. Performance targets

| Metric | Target |
|---|---|
| Render frame rate (M-series Mac) | 60 fps with all stars visible |
| Capture overhead per token | under 50 ms on top of the forward pass |
| Time from Run to first token visible | under 1 second after the model is loaded |
| Cold start (model already downloaded) | under 10 seconds |
| Memory | under 3 GB total |

## 10. Milestones

| # | Milestone | Deliverable | Estimate |
|---|---|---|---|
| 1 | Capture | Script that prints real MLP, attention and residual values for a prompt, with a correctness test | 1 day |
| 2 | Layout | Precompute script producing `layout.bin`; static galaxy renders in the browser | 1 day |
| 3 | Live link | WebSocket streaming; stars light up from real values during generation | 1–2 days |
| 4 | Attention and playback | Beams, token row, stepper, scrubber, next-token panel | 2 days |
| 5 | Trust and polish | Hover cards, raw mode, threshold, bloom tuning, one-command start | 1–2 days |

Total: about one week.

## 11. Success criteria for v0

- A new user gets from clone to a lit galaxy with one install command and one run command.
- The verification test passes: values shown match an independent forward pass within float16 precision.
- Every star and beam is hoverable and shows its source value.
- 60 fps holds during playback on an M-series Mac.
- A 30-second screen recording is understandable without narration.

## 12. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Looks like noise rather than structure | Product is not compelling | Threshold slider, per-layer scaling, show only top-k neurons by default |
| UMAP layout looks like a blob | Weak "galaxy" feel | Try clustering-based colouring and tune UMAP parameters; fall back to 3D layout |
| Users read meaning into individual stars | Misleading | State clearly in the UI that neurons are not individually interpretable in v0 |
| Too many attention beams | Visual clutter | Default to one layer and the strongest heads |
| Hook or attention API changes between `transformers` versions | Capture breaks | Pin versions; keep the capture test in CI |

## 13. Roadmap after v0

1. **More small models:** SmolLM, Qwen 0.5B, Gemma 270M, behind a model adapter interface.
2. **Bigger models:** activation downsampling and level-of-detail rendering for 1B+ parameters.
3. **Interpretable stars:** sparse autoencoder features with labels in place of raw neurons.
4. **Native engine hooks:** capture from llama.cpp or MLX so any local GGUF model works.
5. **Sharing:** export a run as a self-contained replay file or video.
6. **Comparison:** two prompts side by side, with the difference highlighted.

## 14. Open questions

1. Should the layout be 2D discs stacked in 3D (proposed) or a full 3D cloud per layer?
2. Should attention heads be their own objects in the scene, or only appear as beams?
3. Should the default view show all 36,864 stars or only the top few percent by activation?
4. Is a desktop wrapper needed, or is "local server plus browser tab" enough for v0?
5. What is the final name?
