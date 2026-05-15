"""Multimodal-projector outputs captured via hook."""
import torch


def pool_projector_features(projector_output):
    if projector_output is None:
        return None
    feat = projector_output
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    if feat.dim() == 3:
        return feat.mean(dim=1).squeeze(0).detach().cpu().to(torch.float32)
    return feat.detach().cpu().to(torch.float32)


def projector_tokens(projector_output):
    feat = projector_output
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    if feat.dim() == 3:
        return feat.squeeze(0).detach().cpu().to(torch.float32)
    return feat.detach().cpu().to(torch.float32)
