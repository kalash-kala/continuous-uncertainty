"""Yes/no first-token logit extraction and margin/entropy computation."""
import math
import torch
from ..metrics.logit_metrics import logsumexp, binary_entropy_from_logits


def get_yes_no_token_ids(tokenizer, yes_variants, no_variants):
    """Map word variants to first-subword token ids."""
    def variants_to_ids(variants):
        ids = set()
        for w in variants:
            toks = tokenizer.encode(w, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
        return sorted(ids)
    return variants_to_ids(yes_variants), variants_to_ids(no_variants)


def compute_yes_no_logits(outputs, tokenizer, yes_variants, no_variants,
                          first_answer_position=None):
    """Compute (logit_yes, logit_no, margin, binary_entropy, p_yes, parsed)."""
    logits = outputs.logits  # [B, S, V]
    if first_answer_position is None or first_answer_position >= logits.shape[1]:
        first_answer_position = logits.shape[1] - 1
    row = logits[0, first_answer_position].detach().to(torch.float32).cpu()

    yes_ids, no_ids = get_yes_no_token_ids(tokenizer, yes_variants, no_variants)
    if not yes_ids or not no_ids:
        return None

    logit_yes = logsumexp([row[t].item() for t in yes_ids])
    logit_no = logsumexp([row[t].item() for t in no_ids])
    margin = logit_yes - logit_no
    ent, p_yes = binary_entropy_from_logits(logit_yes, logit_no)
    parsed = "yes" if margin > 0 else "no"
    return {
        "logit_yes": logit_yes,
        "logit_no": logit_no,
        "logit_margin_yes_no": margin,
        "binary_entropy": ent,
        "p_yes": p_yes,
        "parsed_answer": parsed,
        "first_answer_position": int(first_answer_position),
    }
