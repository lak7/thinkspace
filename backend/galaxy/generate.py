"""Token-by-token generation that yields one captured Frame per position."""

from __future__ import annotations

from typing import Callable, Iterator

import torch

from .capture import Capture, Frame, make_frame
from .model import MAX_CONTEXT

MAX_NEW_TOKENS = 256


class PromptError(ValueError):
    pass


def sample(logits: torch.Tensor, temperature: float, top_k: int, gen: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = logits.topk(k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


def generate(
    capture: Capture,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 0,
    should_stop: Callable[[], bool] = lambda: False,
) -> Iterator[Frame]:
    device = next(capture.model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    n_prompt = ids.shape[1]
    if n_prompt == 0:
        raise PromptError("The prompt is empty.")
    if n_prompt > MAX_CONTEXT:
        raise PromptError(
            f"The prompt is {n_prompt} tokens; GPT-2's limit is {MAX_CONTEXT}."
        )
    max_new_tokens = max(0, min(max_new_tokens, MAX_NEW_TOKENS, MAX_CONTEXT - n_prompt))
    gen = torch.Generator().manual_seed(seed)

    step = capture.forward(ids.to(device))
    for t in range(n_prompt):
        yield make_frame(step, t, t, int(ids[0, t]), True, tokenizer)

    pos = n_prompt
    for _ in range(max_new_tokens):
        if should_stop():
            return
        token_id = sample(step.logits[-1], temperature, top_k, gen)
        step = capture.forward(torch.tensor([[token_id]], device=device), past=step.past)
        yield make_frame(step, 0, pos, token_id, False, tokenizer)
        pos += 1
