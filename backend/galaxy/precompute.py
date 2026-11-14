"""One-off precompute: star positions and brightness scales.

Positions: two star maps per layer, each a 2D UMAP of one 768-dim vector
per neuron (3,072 neurons).

- "writes" (layout.bin): the neuron's output direction, its row of
  ``mlp.c_proj.weight``. Neurons that write similar directions into the
  residual stream land near each other.
- "reads" (layout_reads.bin): the neuron's input direction, its column of
  ``mlp.c_fc.weight`` with the ln_2 gain folded in and the mean removed
  (LayerNorm already removes it, so it cannot affect firing). Neurons that
  fire on similar inputs land near each other, so co-firing stars clump.

Scales: the 99th percentile of |activation| per layer over a calibration
text, so brightness is comparable across tokens and prompts.

Run with:  uv run python -m galaxy.precompute
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import umap

from .capture import Capture
from .model import MODEL_NAME, load

SEED = 7
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CALIBRATION = Path(__file__).resolve().parent / "calibration.txt"


def layout_layer(weights: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2, metric="cosine", n_neighbors=30, min_dist=0.15, random_state=SEED
    )
    xy = reducer.fit_transform(weights).astype(np.float32)
    xy -= xy.mean(axis=0)
    r = np.linalg.norm(xy, axis=1)
    xy /= np.percentile(r, 99.5)
    # Pull the few outliers back onto the unit circle.
    r = np.linalg.norm(xy, axis=1, keepdims=True)
    return np.where(r > 1, xy / r, xy).astype(np.float32)


def write_vectors(block) -> np.ndarray:
    """(d_mlp, d_model): what each neuron adds to the residual stream per unit activation."""
    return block.mlp.c_proj.weight.detach().cpu().numpy()


def read_vectors(block) -> np.ndarray:
    """(d_mlp, d_model): the direction in the normalised residual each neuron fires on.

    pre_act_i = ln_2(x) . w_i + b_i, and ln_2(x) = g * n(x) + beta with n(x)
    zero-mean, so pre_act_i = n(x) . (g * w_i - mean) + const.
    """
    w = block.mlp.c_fc.weight.detach().cpu().numpy().T  # Conv1D stores (in, out)
    w = w * block.ln_2.weight.detach().cpu().numpy()[None, :]
    return w - w.mean(axis=1, keepdims=True)


LAYOUTS = {
    "writes": ("layout.bin", write_vectors, "UMAP (cosine, 2D) of mlp.c_proj.weight rows, per layer"),
    "reads": (
        "layout_reads.bin",
        read_vectors,
        "UMAP (cosine, 2D) of mlp.c_fc.weight columns x ln_2 gain, mean removed, per layer",
    ),
}


def calibrate(capture: Capture, tokenizer, chunk: int = 256) -> list[float]:
    ids = tokenizer(CALIBRATION.read_text(), return_tensors="pt").input_ids[0]
    device = next(capture.model.parameters()).device
    per_layer: list[list[np.ndarray]] = [[] for _ in range(capture.n_layer)]
    for start in range(0, len(ids), chunk):
        step = capture.forward(ids[start : start + chunk].unsqueeze(0).to(device))
        for layer in range(capture.n_layer):
            per_layer[layer].append(step.mlp[layer].abs().flatten().numpy())
    return [float(np.percentile(np.concatenate(v), 99)) for v in per_layer]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    model, tokenizer = load(torch.device("cpu"))
    cfg = model.config
    d_mlp = model.transformer.h[0].mlp.c_proj.weight.shape[0]

    layouts = {}
    for name, (filename, vectors, rule) in LAYOUTS.items():
        layout = np.zeros((cfg.n_layer, d_mlp, 2), dtype=np.float32)
        for layer, block in enumerate(model.transformer.h):
            w = vectors(block)
            assert w.shape == (d_mlp, cfg.n_embd), w.shape
            t0 = time.time()
            layout[layer] = layout_layer(w)
            print(f"{name:6s} layer {layer + 1:2d}/{cfg.n_layer}  UMAP {time.time() - t0:.1f}s")
        layout.astype("<f4").tofile(DATA_DIR / filename)
        layouts[name] = {"file": filename, "rule": rule}

    capture = Capture(model)
    scales = calibrate(capture, tokenizer)
    capture.close()
    print("scales:", [round(s, 3) for s in scales])

    manifest = {
        "model": MODEL_NAME,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "d_model": cfg.n_embd,
        "d_mlp": d_mlp,
        "scales": scales,
        "scale_rule": "99th percentile of |activation| per layer over calibration.txt",
        "layout_rule": LAYOUTS["writes"][2],
        "layouts": layouts,
        "layout_seed": SEED,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "umap": umap.__version__,
        },
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {', '.join(v['file'] for v in layouts.values())} and manifest.json to {DATA_DIR}")


if __name__ == "__main__":
    main()
