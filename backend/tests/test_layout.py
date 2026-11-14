"""The vectors the star maps are built from mean what the README says they mean."""

import torch

from galaxy.precompute import read_vectors, write_vectors


@torch.inference_mode()
def test_read_vectors_drive_firing(model_and_tokenizer):
    """Each neuron's pre-activation is n(x) . read_vector + a per-neuron constant,
    where n(x) is the residual after LayerNorm's centring and scaling (no gain)."""
    model, tokenizer = model_and_tokenizer
    ids = tokenizer("The capital of France is Paris, and", return_tensors="pt").input_ids
    hidden = model(ids, output_hidden_states=True).hidden_states
    for layer in (0, 5, 11):
        block = model.transformer.h[layer]
        # Input to ln_2 is the residual after this block's attention.
        x = hidden[layer][0]
        x = x + block.attn(block.ln_1(x).unsqueeze(0))[0][0]
        pre = block.mlp.c_fc(block.ln_2(x))  # (T, d_mlp)
        n = torch.nn.functional.layer_norm(x, x.shape[-1:], eps=block.ln_2.eps)
        r = torch.from_numpy(read_vectors(block))
        const = pre - n @ r.T
        assert torch.allclose(const, const[0].expand_as(const), atol=1e-3), layer


@torch.inference_mode()
def test_write_vectors_are_mlp_output(model_and_tokenizer):
    model, _ = model_and_tokenizer
    block = model.transformer.h[3]
    act = torch.randn(4, block.mlp.c_proj.weight.shape[0])
    w = torch.from_numpy(write_vectors(block))
    assert torch.allclose(block.mlp.c_proj(act) - block.mlp.c_proj.bias, act @ w, atol=1e-4)
