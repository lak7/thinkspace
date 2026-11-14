import numpy as np

from galaxy.capture import Frame
from galaxy.protocol import decode_frame, encode_frame


def test_round_trip():
    rng = np.random.default_rng(0)
    for seq_len, token in [(1, "a"), (7, " café"), (130, "\n")]:
        frame = Frame(
            pos=seq_len - 1,
            token_id=42,
            token=token,
            is_prompt=False,
            mlp=rng.normal(size=(12, 3072)).astype(np.float32),
            attn=rng.random(size=(12, 12, seq_len)).astype(np.float32),
            resid=rng.random(13).astype(np.float32),
            top=[(i, f"t{i}", 0.1) for i in range(10)],
        )
        header, mlp, attn = decode_frame(encode_frame(frame))
        assert header["pos"] == seq_len - 1
        assert header["token"] == token
        assert header["seq_len"] == seq_len
        assert len(header["top"]) == 10
        np.testing.assert_array_equal(mlp, frame.mlp.astype(np.float16))
        np.testing.assert_array_equal(attn, frame.attn.astype(np.float16))
        np.testing.assert_allclose(header["resid"], frame.resid, rtol=1e-6)
