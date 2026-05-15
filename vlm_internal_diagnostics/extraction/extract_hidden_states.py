"""Layerwise hidden-state pooling at final_prompt / image_mean / question_mean."""
import torch


def collect_hidden_states(outputs, token_indices, layers="all"):
    """Returns dict: source -> dict {layer -> 1D tensor}."""
    hs = outputs.hidden_states  # tuple len = num_layers+1
    num_layers = len(hs)
    if layers == "all":
        layer_iter = range(num_layers)
    else:
        layer_iter = [l for l in layers if l < num_layers]

    final_idx = token_indices["final_prompt_token_index"]
    img_idx = token_indices["image_token_indices"]
    q_idx = token_indices["question_token_indices"]

    out = {"final_prompt": {}, "image_mean": {}, "question_mean": {}}
    for l in layer_iter:
        h = hs[l][0]  # [seq, dim]
        seq_len = h.shape[0]
        fi = min(final_idx, seq_len - 1)
        out["final_prompt"][l] = h[fi].detach().cpu().to(torch.float32)
        if img_idx:
            ii = [i for i in img_idx if i < seq_len]
            if ii:
                out["image_mean"][l] = h[ii].mean(dim=0).detach().cpu().to(torch.float32)
        if q_idx:
            qi = [i for i in q_idx if i < seq_len]
            if qi:
                out["question_mean"][l] = h[qi].mean(dim=0).detach().cpu().to(torch.float32)
    return out


def hidden_state_norms(hidden_by_source_layer):
    norms = {}
    for src, by_layer in hidden_by_source_layer.items():
        for layer, vec in by_layer.items():
            norms[f"layer_{layer}_{src}"] = float(vec.norm().item())
    return norms
