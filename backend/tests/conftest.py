import pytest
import torch

from galaxy.model import load


@pytest.fixture(scope="session")
def model_and_tokenizer():
    return load(torch.device("cpu"))


@torch.inference_mode()
def reference_pass(model, ids: torch.Tensor):
    """Independent, uncached forward pass over a whole sequence.

    Deliberately uses different hook points from galaxy.capture: MLP
    activations are read from the *output* of each block's activation module
    (capture reads the *input* of c_proj), pre-activations from its *input*
    (capture reads the *output* of c_fc), and residuals come from
    output_hidden_states.
    """
    acts: list[torch.Tensor] = []
    pres: list[torch.Tensor] = []
    def hook(_m, args, out) -> None:  # must return None, or it replaces the output
        pres.append(args[0].detach())
        acts.append(out.detach())

    handles = [block.mlp.act.register_forward_hook(hook) for block in model.transformer.h]
    try:
        out = model(ids, use_cache=False, output_attentions=True, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    mlp = torch.stack([a[0] for a in acts])  # (L, T, d_mlp)
    pre = torch.stack([a[0] for a in pres])  # (L, T, d_mlp)
    attn = torch.stack([a[0] for a in out.attentions])  # (L, H, T, T)
    # hidden_states[0..L-1] are the inputs to each block, i.e. embedding output
    # and the outputs of blocks 1..L-1. The last entry is post-ln_f, so it is
    # not comparable and is left out.
    resid = torch.stack([h[0] for h in out.hidden_states[:-1]]).norm(dim=-1)  # (L, T)
    return mlp, attn, resid, out.logits[0], pre
