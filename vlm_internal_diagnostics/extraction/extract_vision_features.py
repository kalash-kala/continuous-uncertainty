"""Vision-encoder feature extraction. Called via hook during forward."""
import torch


def pool_vision_features(vision_output):
    """Mean-pool the vision encoder last hidden state over patch tokens."""
    if vision_output is None:
        return None
    # CLIPVisionModelOutput-like has last_hidden_state.
    feat = getattr(vision_output, "last_hidden_state", None)
    if feat is None and isinstance(vision_output, (tuple, list)):
        feat = vision_output[0]
    if feat is None:
        return None
    return feat.mean(dim=1).squeeze(0).detach().cpu().to(torch.float32)


def patch_features(vision_output):
    """Return per-patch features (drops CLS token if applicable)."""
    feat = getattr(vision_output, "last_hidden_state", None)
    if feat is None and isinstance(vision_output, (tuple, list)):
        feat = vision_output[0]
    if feat is None:
        return None
    # Drop CLS for default LLaVA-1.5 strategy.
    return feat[:, 1:, :].squeeze(0).detach().cpu().to(torch.float32)
