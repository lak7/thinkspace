"""Binary wire format for one Frame.

    uint32 LE   header length in bytes (N)
    N bytes     UTF-8 JSON header
    0-3 bytes   zero padding so the payload starts on a 4-byte boundary
    float16 LE  MLP activations (post-GELU), n_layer * d_mlp values, layer-major
    float16 LE  MLP pre-activations (pre-GELU), n_layer * d_mlp values, layer-major
    float16 LE  attention weights, n_layer * n_head * seq_len values
"""

from __future__ import annotations

import json
import struct

import numpy as np

from .capture import Frame

VERSION = 2


def encode_frame(frame: Frame) -> bytes:
    n_layer, d_mlp = frame.mlp.shape
    _, n_head, seq_len = frame.attn.shape
    header = {
        "v": VERSION,
        "pos": frame.pos,
        "token_id": frame.token_id,
        "token": frame.token,
        "is_prompt": frame.is_prompt,
        "seq_len": seq_len,
        "n_layer": n_layer,
        "n_head": n_head,
        "d_mlp": d_mlp,
        "resid": [float(x) for x in frame.resid],
        "top": [{"id": i, "token": t, "p": p} for i, t, p in frame.top],
    }
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(4 + len(hb))) % 4
    return b"".join(
        [
            struct.pack("<I", len(hb)),
            hb,
            b"\0" * pad,
            frame.mlp.astype("<f2").tobytes(),
            frame.pre.astype("<f2").tobytes(),
            frame.attn.astype("<f2").tobytes(),
        ]
    )


def decode_frame(data: bytes) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of encode_frame. Returns (header, mlp, pre, attn) with float16 arrays."""
    (hlen,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4 : 4 + hlen].decode("utf-8"))
    off = 4 + hlen
    off += (-off) % 4
    n_mlp = header["n_layer"] * header["d_mlp"]
    mlp = np.frombuffer(data, dtype="<f2", count=n_mlp, offset=off).reshape(
        header["n_layer"], header["d_mlp"]
    )
    off += n_mlp * 2
    pre = np.frombuffer(data, dtype="<f2", count=n_mlp, offset=off).reshape(
        header["n_layer"], header["d_mlp"]
    )
    off += n_mlp * 2
    n_attn = header["n_layer"] * header["n_head"] * header["seq_len"]
    attn = np.frombuffer(data, dtype="<f2", count=n_attn, offset=off).reshape(
        header["n_layer"], header["n_head"], header["seq_len"]
    )
    assert off + n_attn * 2 == len(data), "trailing bytes in frame"
    return header, mlp, pre, attn
