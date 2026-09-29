"""Model adapter pattern: abstract per-model behavior behind a registry.

Each adapter implements the small interface needed by extraction/extract_all.py:
  - load(model_cfg) -> (model, processor)
  - build_prompt(processor, question, prompt_cfg) -> str
  - preprocess_image(image) -> PIL.Image  (forces fixed input resolution)
  - get_token_indices(processor, model, image, question, prompt) -> dict
  - capture_vision_and_projector(model) -> contextmanager yielding {"vision":..., "projector":...}
  - pool_vision_features(hook_output) -> torch.Tensor or None
  - pool_projector_features(hook_output) -> torch.Tensor or None

Each adapter declares:
  family: short string used in model_config.yaml
  fixed_input_size: int side length (square) the image is resized to
  expected_num_patches: image-token count after the projector
  expected_grid_shape: (rows, cols)
"""
from typing import Dict, Type


class ModelAdapter:
    family: str = ""
    fixed_input_size: int = 0
    expected_num_patches: int = 0
    expected_grid_shape: tuple = (0, 0)

    def load(self, model_cfg):
        raise NotImplementedError

    def build_prompt(self, processor, question, prompt_cfg):
        raise NotImplementedError

    def preprocess_image(self, image):
        """Default: resize to (fixed_input_size, fixed_input_size) with bicubic."""
        if not self.fixed_input_size:
            return image
        return image.resize((self.fixed_input_size, self.fixed_input_size), resample=3)

    def get_token_indices(self, processor, model, image, question, prompt, generated_ids=None):
        raise NotImplementedError

    def capture_vision_and_projector(self, model):
        raise NotImplementedError

    def pool_vision_features(self, vision_output):
        raise NotImplementedError

    def pool_projector_features(self, projector_output):
        raise NotImplementedError


_REGISTRY: Dict[str, Type[ModelAdapter]] = {}


def register_adapter(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_adapter(family: str) -> ModelAdapter:
    if family not in _REGISTRY:
        # Lazy-import known adapters so we only pay the import cost for ones we use.
        _maybe_import(family)
    if family not in _REGISTRY:
        raise ValueError(
            f"Unknown model family '{family}'. Known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[family]()


def _maybe_import(family: str):
    try:
        if family == "llava":
            from . import llava  # noqa: F401
        elif family == "qwen2_5_vl":
            from . import qwen25_vl  # noqa: F401
        elif family == "phi4_multimodal":
            from . import phi4_multimodal  # noqa: F401
        elif family == "pixtral":
            from . import pixtral  # noqa: F401
    except Exception as e:
        print(f"[adapters] failed to import adapter for '{family}': {e}")
