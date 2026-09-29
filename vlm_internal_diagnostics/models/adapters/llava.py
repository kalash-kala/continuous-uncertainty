"""LLaVA-1.5 adapter. Thin wrapper around the existing llava_loader / token_index_utils."""
from contextlib import contextmanager

from . import ModelAdapter, register_adapter
from ..llava_loader import load_llava, build_prompt
from ..token_index_utils import get_token_indices
from ..hook_utils import capture_vision_and_projector
from ...extraction.extract_vision_features import pool_vision_features
from ...extraction.extract_projector_features import pool_projector_features


@register_adapter("llava")
class LlavaAdapter(ModelAdapter):
    family = "llava"
    fixed_input_size = 336
    expected_num_patches = 576
    expected_grid_shape = (24, 24)

    def load(self, model_cfg):
        return load_llava(model_cfg)

    def build_prompt(self, processor, question, prompt_cfg):
        return build_prompt(processor, question, prompt_cfg)

    def get_token_indices(self, processor, model, image, question, prompt, generated_ids=None):
        return get_token_indices(processor, model, image, question, prompt, generated_ids)

    @contextmanager
    def capture_vision_and_projector(self, model):
        with capture_vision_and_projector(model) as store:
            yield store

    def pool_vision_features(self, vision_output):
        return pool_vision_features(vision_output)

    def pool_projector_features(self, projector_output):
        return pool_projector_features(projector_output)
