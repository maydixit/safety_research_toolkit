"""
Token-level perplexity computation for language models.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

IGNORE_INDEX = -100


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    batch_size: int = 8,
    device: Optional[str] = None,
    assistant_header: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """Compute token-level perplexity over a list of text strings.

    If *assistant_header* is provided the loss is masked to only assistant
    tokens (all tokens between each occurrence of *assistant_header* and the
    next EOS token are scored). Otherwise all tokens are scored.

    Args:
        model: Causal LM (already loaded, ``eval()`` recommended).
        tokenizer: Corresponding tokenizer.
        texts: List of plain-text strings to evaluate.
        batch_size: Texts per forward pass.
        device: Target device (default: first model parameter's device).
        assistant_header: Token sequence marking the start of assistant turns.
            Pass ``None`` to score all tokens.
        max_length: Truncate sequences longer than this (default: model max).

    Returns:
        ``(aggregate_ppl, per_example)`` where ``per_example`` is a list of
        ``(orig_idx, n_scored_tokens, ppl)`` tuples.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    max_len = max_length or model.config.max_position_embeddings
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    header_ids = None
    header_len = 0
    eot_id = tokenizer.eos_token_id
    if assistant_header is not None:
        header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
        header_len = len(header_ids)

    examples: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    for idx, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_len:
            continue

        if header_ids is not None:
            labels = [IGNORE_INDEX] * len(ids)
            i = 0
            while i <= len(ids) - header_len:
                if ids[i : i + header_len] == header_ids:
                    start = i + header_len
                    end = start
                    while end < len(ids) and ids[end] != eot_id:
                        end += 1
                    for k in range(start, end):
                        labels[k] = ids[k]
                    i = end
                else:
                    i += 1
            if not any(l != IGNORE_INDEX for l in labels):
                continue
        else:
            labels = list(ids)

        examples.append((idx, torch.tensor(ids), torch.tensor(labels)))

    if not examples:
        return float("nan"), []

    examples.sort(key=lambda x: x[1].size(0))

    total_nll, total_tokens = 0.0, 0
    per_example: List[Tuple[int, int, float]] = []

    for b_start in range(0, len(examples), batch_size):
        batch = examples[b_start : b_start + batch_size]
        orig_indices = [e[0] for e in batch]

        input_ids = pad_sequence(
            [e[1] for e in batch], batch_first=True, padding_value=pad_id
        ).to(device)
        labels_t = pad_sequence(
            [e[2] for e in batch], batch_first=True, padding_value=IGNORE_INDEX
        ).to(device)
        # Mask by sequence length, not token value: the pad id is often the
        # EOS / end-of-turn id, which also appears inside the real text.
        attention_mask = pad_sequence(
            [torch.ones_like(e[1]) for e in batch], batch_first=True, padding_value=0
        ).to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        tgt = labels_t[:, 1:]

        per_tok = cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(tgt.size())

        mask = tgt != IGNORE_INDEX
        for i, orig_idx in enumerate(orig_indices):
            n_tok = int(mask[i].sum().item())
            if n_tok == 0:
                continue
            nll_i = float(per_tok[i][mask[i]].sum().item())
            ppl_i = float(np.exp(nll_i / n_tok))
            total_nll += nll_i
            total_tokens += n_tok
            per_example.append((orig_idx, n_tok, ppl_i))

    agg_ppl = float(np.exp(total_nll / total_tokens)) if total_tokens else float("nan")
    return agg_ppl, per_example
