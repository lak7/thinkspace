"""The values we show must equal what the model computes.

Frames are produced the way the app produces them (prompt pass, then
token-by-token with a KV cache). They are compared against one uncached
forward pass over the final sequence, read through different hook points.
"""

import numpy as np
import pytest
import torch
from conftest import reference_pass

from galaxy.capture import Capture
from galaxy.generate import PromptError, generate

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def run(model_and_tokenizer):
    model, tok = model_and_tokenizer
    cap = Capture(model)
    frames = list(generate(cap, tok, PROMPT, max_new_tokens=12, temperature=0.8, top_k=40, seed=3))
    cap.close()
    ids = torch.tensor([[f.token_id for f in frames]])
    return frames, reference_pass(model, ids)


def test_shapes_and_order(run, model_and_tokenizer):
    frames, _ = run
    _, tok = model_and_tokenizer
    n_prompt = len(tok(PROMPT).input_ids)
    assert len(frames) == n_prompt + 12
    for i, f in enumerate(frames):
        assert f.pos == i
        assert f.is_prompt == (i < n_prompt)
        assert f.mlp.shape == (12, 3072)
        assert f.attn.shape == (12, 12, i + 1)
        assert f.resid.shape == (13,)
        assert len(f.top) == 10


def test_mlp_matches_reference(run):
    frames, (mlp, _, _, _) = run
    for f in frames:
        np.testing.assert_allclose(f.mlp, mlp[:, f.pos].numpy(), atol=2e-3, rtol=1e-3)


def test_attention_matches_reference(run):
    frames, (_, attn, _, _) = run
    for f in frames:
        ref = attn[:, :, f.pos, : f.pos + 1].numpy()
        np.testing.assert_allclose(f.attn, ref, atol=1e-4)
        np.testing.assert_allclose(f.attn.sum(-1), 1.0, atol=1e-4)


def test_residual_matches_reference(run):
    frames, (_, _, resid, _) = run
    for f in frames:
        np.testing.assert_allclose(f.resid[:12], resid[:, f.pos].numpy(), rtol=1e-3)


def test_candidates_match_reference(run):
    frames, (_, _, _, logits) = run
    for f in frames:
        probs = torch.softmax(logits[f.pos], -1)
        p, idx = probs.topk(10)
        assert [t[0] for t in f.top][:3] == [int(i) for i in idx[:3]]
        np.testing.assert_allclose([t[2] for t in f.top], p.numpy(), atol=1e-4)


def test_seed_is_reproducible(model_and_tokenizer):
    model, tok = model_and_tokenizer
    cap = Capture(model)
    a = [f.token_id for f in generate(cap, tok, PROMPT, max_new_tokens=8, seed=11)]
    b = [f.token_id for f in generate(cap, tok, PROMPT, max_new_tokens=8, seed=11)]
    cap.close()
    assert a == b


def test_prompt_limits(model_and_tokenizer):
    model, tok = model_and_tokenizer
    cap = Capture(model)
    with pytest.raises(PromptError):
        list(generate(cap, tok, "", max_new_tokens=1))
    with pytest.raises(PromptError):
        list(generate(cap, tok, "hello " * 1100, max_new_tokens=1))
    cap.close()
