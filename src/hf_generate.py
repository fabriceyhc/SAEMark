"""Direct HuggingFace text generation — drop-in replacement for SAEMark's TGI client.

SAEMark's utils.generate() talks to a TGI server using the `text_generation` client
library, which supports a `best_of` parameter that returns N independent samples.
This module replicates that behaviour using transformers.AutoModelForCausalLM directly,
avoiding the need for any server process.

The public API is a single function:
    generate_batch(prompt, n, max_new_tokens, temperature) -> list[str]

A module-level singleton holds the loaded model/tokenizer so the first call pays the
load cost and subsequent calls are cheap.  Call init_generator() explicitly if you want
to control which model and device are used before the first generate_batch() call.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None
_model_path: str | None = None


def init_generator(model_path: str, device: str = "cuda:0") -> None:
    """Load (or reload) the generation model onto `device`."""
    global _model, _tokenizer, _model_path

    if _model is not None and _model_path == model_path:
        return  # already loaded

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _tokenizer.padding_side = "left"
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    _model.eval()
    _model_path = model_path


def generate_batch(
    prompt: str,
    n: int = 50,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """Return up to `n` independent completions for `prompt`.

    Only the newly generated tokens are returned (prompt tokens stripped).
    Fewer than `n` strings may be returned if the model hits EOS early for
    some sequences.
    """
    if _model is None:
        raise RuntimeError("Call init_generator(model_path) before generate_batch().")

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            pad_token_id=_tokenizer.eos_token_id,
        )

    completions: list[str] = []
    for seq in outputs:
        new_tokens = seq[input_len:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if text:
            completions.append(text)

    return completions
