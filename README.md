# LLM Galaxy

Watch GPT-2 small think, on your own machine. Every star is one real MLP neuron,
every beam is one real attention weight, and nothing on screen is simulated.

See `prd.md` for the full product spec.

## Run it

Requires `uv`, Node 20+ and `pnpm`.

```sh
make setup   # once: installs deps, computes the star maps (~4 min), builds the UI
make start   # then open http://127.0.0.1:8000
```

Type a prompt and press **Run** (or Cmd/Ctrl+Enter).

## What you are looking at

| On screen | Real source |
|---|---|
| Star position | UMAP per layer of either the neuron's output weights (`mlp.c_proj.weight` row: *what it writes*) or its input weights (`mlp.c_fc.weight` column × `ln_2` gain: *what it reads*). Toggle with **Arrange stars by** |
| Star brightness | abs(value) / layer scale, for the current token. The value is the post-GELU activation by default, or the pre-GELU score (toggle with **Light stars by**) |
| Star colour | Sign of that value: gold positive, blue negative. After GELU a neuron can't go below about −0.17, so in the default view blue is always faint and a neuron pushed hard *off* looks dark, the same as an idle one. Switch to the pre-GELU score to see those as bright blue |
| Beam | Attention weight from the current token to an earlier one; colour = head, intensity = weight |
| Spike above the current token | Attention the token pays to itself |
| Ring around a layer | Residual-stream norm after that block, relative to the run's largest |
| Next-token panel | The model's top-10 probabilities at temperature 1 |

The layer scale is the 99th percentile of abs(value) in that layer over
`backend/galaxy/calibration.txt` (one set for the activation, one for the
score), so brightness is comparable across tokens. The star hover always shows
both the score and the activation.

Playback is a slowed replay of recorded values: layers light up 1 to 12 for
each token. **Raw mode** turns off bloom and fades, leaving only data-driven
pixels. Hover any star, beam or token for its exact value.

## Controls

- **Space** play/pause · **← →** previous/next token · **↓ ↑** previous/next layer
- Click a token in the strip or on the ring to jump to it
- **Attention layer**: *Follow* shows the layer currently lighting up; a number pins one layer
- **Heads**: toggle which of the 12 heads draw beams
- **Arrange stars by**: *What they write* groups neurons by the direction they
  push the residual stream; *What they read* groups them by what makes them
  fire. Brightness is driven by what a neuron reads, so in the *read* layout
  neurons that fire together sit together and light up as visible clumps
- **Light stars by**: *Output (after GELU)* is what the neuron passes on;
  *Score (before GELU)* is how hard the input pushed it on or off

## How it stays honest

`make test` runs the model the way the app does (prompt pass, then token by
token with a KV cache), and compares every MLP activation and pre-activation,
attention weight and residual norm against a separate uncached forward pass read through different
hook points. A second test does the same for the bytes a WebSocket client
actually receives.

## Layout

```
backend/galaxy/   model.py capture.py generate.py protocol.py precompute.py server.py
backend/tests/    capture equivalence, protocol round trip, WebSocket end to end
frontend/src/     main.ts run.ts socket.ts protocol.ts scene/ ui/
```

## Limits in v0

- GPT-2 small only, and one run at a time.
- Individual neurons are mostly not interpretable; read patterns, not single stars.
- Values travel as float16 (about 3 significant digits).
- GPT-2's first token position has a much larger residual norm than the rest,
  so it is excluded from the ring scale and its bars are drawn clipped.
