"""End to end: what a WebSocket client receives equals a fresh forward pass."""

import numpy as np
import torch
from conftest import reference_pass
from fastapi.testclient import TestClient

from galaxy import server
from galaxy.protocol import decode_frame


def _collect(ws):
    frames, final = [], None
    while final is None:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            frames.append(decode_frame(msg["bytes"]))
        else:
            import json

            data = json.loads(msg["text"])
            if data["type"] in ("done", "error"):
                final = data
    return frames, final


def test_stream_matches_fresh_forward_pass():
    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "run",
                "prompt": "Once upon a time",
                "max_new_tokens": 10,
                "temperature": 0.7,
                "top_k": 50,
                "seed": 5,
            }
        )
        frames, final = _collect(ws)
        assert final["type"] == "done", final
        assert final["frames"] == len(frames) == 4 + 10

        model = server.state["capture"].model
        ids = torch.tensor([[h["token_id"] for h, _, _ in frames]])
        mlp, attn, _, _ = reference_pass(model, ids)
        for header, f_mlp, f_attn in frames:
            pos = header["pos"]
            # float16 keeps about 3 significant digits.
            np.testing.assert_allclose(
                f_mlp.astype(np.float32), mlp[:, pos].numpy(), atol=3e-3, rtol=2e-3
            )
            np.testing.assert_allclose(
                f_attn.astype(np.float32), attn[:, :, pos, : pos + 1].numpy(), atol=1e-3
            )


def test_errors_are_reported():
    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "run", "prompt": ""})
        _, final = _collect(ws)
        assert final["type"] == "error"
        ws.send_json({"type": "run", "prompt": "word " * 1100})
        _, final = _collect(ws)
        assert final["type"] == "error" and "1024" in final["message"]
