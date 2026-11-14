"""One-off precompute: star positions and brightness scales.

Positions: for each layer, the 3,072 neuron output directions (rows of
``mlp.c_proj.weight``, 768 dims each) are reduced to 2D with UMAP. Neurons
that write similar directions into the residual stream land near each other.

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

    layout = np.zeros((cfg.n_layer, d_mlp, 2), dtype=np.float32)
    for layer, block in enumerate(model.transformer.h):
        w = block.mlp.c_proj.weight.detach().cpu().numpy()
        assert w.shape == (d_mlp, cfg.n_embd), w.shape
        t0 = time.time()
        layout[layer] = layout_layer(w)
        print(f"layer {layer + 1:2d}/{cfg.n_layer}  UMAP {time.time() - t0:.1f}s")

    capture = Capture(model)
    scales = calibrate(capture, tokenizer)
    capture.close()
    print("scales:", [round(s, 3) for s in scales])

    layout.astype("<f4").tofile(DATA_DIR / "layout.bin")
    manifest = {
        "model": MODEL_NAME,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "d_model": cfg.n_embd,
        "d_mlp": d_mlp,
        "scales": scales,
        "scale_rule": "99th percentile of |activation| per layer over calibration.txt",
        "layout_rule": "UMAP (cosine, 2D) of mlp.c_proj.weight rows, per layer",
        "layout_seed": SEED,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "umap": umap.__version__,
        },
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {DATA_DIR / 'layout.bin'} and manifest.json")


if __name__ == "__main__":
    main()
