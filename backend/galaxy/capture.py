"""Read real internal values out of a GPT-2 forward pass.

Three things are captured, all for the positions processed in the call:

- MLP activations: the input to each block's ``mlp.c_proj``, which is exactly
  the post-GELU hidden activation (3,072 values per layer).
- Attention weights: returned by the model itself (``output_attentions``).
- Residual stream: the embedding output and the output of every block,
  before the final layer norm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Step:
    """Raw tensors from one forward call over ``T_new`` positions."""

    mlp: torch.Tensor  # (n_layer, T_new, d_mlp)
    attn: torch.Tensor  # (n_layer, n_head, T_new, T_total)
    resid: torch.Tensor  # (n_layer + 1, T_new)  L2 norms
    logits: torch.Tensor  # (T_new, vocab)
    past: object  # KV cache to pass to the next call


@dataclass
class Frame:
    """Everything shown for one token position."""

    pos: int
    token_id: int
    token: str
    is_prompt: bool
    mlp: np.ndarray  # (n_layer, d_mlp) float32
    attn: np.ndarray  # (n_layer, n_head, pos + 1) float32
    resid: np.ndarray  # (n_layer + 1,) float32
    top: list[tuple[int, str, float]]  # next-token candidates at temperature 1


class Capture:
    def __init__(self, model):
        self.model = model
        self.n_layer = model.config.n_layer
        self._mlp: list[torch.Tensor | None] = [None] * self.n_layer
        self._block_out: list[torch.Tensor | None] = [None] * self.n_layer
        self._handles = []
        for i, block in enumerate(model.transformer.h):
            self._handles.append(
                block.mlp.c_proj.register_forward_pre_hook(self._mlp_hook(i))
            )
            self._handles.append(block.register_forward_hook(self._block_hook(i)))

    def _mlp_hook(self, i: int):
        def hook(_module, args):
            self._mlp[i] = args[0].detach()

        return hook

    def _block_hook(self, i: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self._block_out[i] = hidden.detach()

        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, past=None) -> Step:
        out = self.model(
            input_ids,
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        mlp = torch.stack([m[0] for m in self._mlp])
        attn = torch.stack([a[0] for a in out.attentions])
        embed = out.hidden_states[0][0]
        resid = torch.stack([embed] + [b[0] for b in self._block_out]).norm(dim=-1)
        return Step(
            mlp=mlp.float().cpu(),
            attn=attn.float().cpu(),
            resid=resid.float().cpu(),
            logits=out.logits[0].float().cpu(),
            past=out.past_key_values,
        )


def make_frame(
    step: Step, t: int, pos: int, token_id: int, is_prompt: bool, tokenizer, top_n: int = 10
) -> Frame:
    """Slice position ``t`` of ``step`` (absolute position ``pos``) into a Frame."""
    probs = torch.softmax(step.logits[t], dim=-1)
    p, idx = probs.topk(top_n)
    top = [(int(i), tokenizer.decode([int(i)]), float(v)) for i, v in zip(idx, p)]
    return Frame(
        pos=pos,
        token_id=token_id,
        token=tokenizer.decode([token_id]),
        is_prompt=is_prompt,
        mlp=step.mlp[:, t].numpy().copy(),
        attn=step.attn[:, :, t, : pos + 1].numpy().copy(),
        resid=step.resid[:, t].numpy().copy(),
        top=top,
    )
