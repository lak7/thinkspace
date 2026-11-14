"""Load GPT-2 small in a form that exposes its internals."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_NAME = "gpt2"
MAX_CONTEXT = 1024


def pick_device() -> torch.device:
    """GALAXY_DEVICE overrides; otherwise CPU.

    GPT-2 small with a KV cache runs single-token steps faster on an M-series
    CPU than on MPS, and CPU keeps the cached and uncached paths numerically
    closest, which the verification tests rely on.
    """
    name = os.environ.get("GALAXY_DEVICE", "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    return torch.device(name)


def load(device: torch.device | None = None):
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Eager attention is required: the optimised attention path does not
    # return attention weights.
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer
