"""Pixtral-12B-2409 adapter.

Notes:
  - Image is forced to a fixed 1024x1024 square via adapter.preprocess_image.
    With Pixtral's 16x16 patch size this yields 4096 image tokens
    (64x64 grid) in the LM sequence.
  - Hooks: vision_tower for vision-encoder features; multi_modal_projector
    for the projected features that the LM sees.
  - attn_implementation forced to "eager" so output_attentions returns tensors.
"""
from contextlib import contextmanager
import torch

from . import ModelAdapter, register_adapter
from ..token_index_utils import build_token_indices_from_ids


_DTYPE_MAP = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


@register_adapter("pixtral")
class PixtralAdapter(ModelAdapter):
    family = "pixtral"
    fixed_input_size = 512
    expected_num_patches = 1024  # 32 x 32
    expected_grid_shape = (32, 32)

    def load(self, model_cfg):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        name = model_cfg["name"]
        dtype = _DTYPE_MAP.get(str(model_cfg.get("torch_dtype", "bfloat16")).lower(),
                               torch.bfloat16)
        attn_impl = "eager"

        # Pixtral uses Mistral's tokenizer format; PixtralProcessor is needed
        # (transformers >= 4.45). Fall back to AutoProcessor for other variants.
        try:
            from transformers import PixtralProcessor
            processor = PixtralProcessor.from_pretrained(
                name, size={"longest_edge": self.fixed_input_size},
            )
        except (ImportError, Exception):
            processor = AutoProcessor.from_pretrained(
                name, size={"longest_edge": self.fixed_input_size},
            )

        kwargs = dict(
            torch_dtype=dtype,
            device_map=model_cfg.get("device_map", "auto"),
            attn_implementation=attn_impl,
        )
        if model_cfg.get("load_in_4bit"):
            kwargs["load_in_4bit"] = True
        if model_cfg.get("load_in_8bit"):
            kwargs["load_in_8bit"] = True

        model = LlavaForConditionalGeneration.from_pretrained(name, **kwargs)
        try:
            model.config._attn_implementation = "eager"
            if hasattr(model.config, "text_config"):
                model.config.text_config._attn_implementation = "eager"
            if hasattr(model.config, "vision_config"):
                model.config.vision_config._attn_implementation = "eager"
        except Exception:
            pass
        model.eval()
        return model, processor

    def build_prompt(self, processor, question, prompt_cfg):
        # Pixtral instruction format: [INST]\n[IMG]\n{question}[/INST]
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }]
        try:
            return processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            return prompt_cfg.get(
                "fallback_template",
                "<s>[INST][IMG]\n{question}[/INST]",
            ).format(question=question)

    def get_token_indices(self, processor, model, image, question, prompt, generated_ids=None):
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        img_tok = None
        for attr in ("image_token_index", "image_token_id"):
            v = getattr(model.config, attr, None)
            if v is not None:
                img_tok = int(v); break
        if img_tok is None:
            try:
                img_tok = processor.tokenizer.convert_tokens_to_ids("[IMG]")
            except Exception:
                img_tok = None
        # Pixtral processor pre-expands [IMG] to per-patch tokens — detect actual count
        if img_tok is not None:
            n_img = int((input_ids == img_tok).sum())
            num_img = n_img if n_img > 0 else self.expected_num_patches
        else:
            num_img = self.expected_num_patches
        return build_token_indices_from_ids(
            processor, input_ids,
            inputs.get("attention_mask"), inputs.get("pixel_values"),
            image_token_id=img_tok,
            num_image_tokens=num_img,
            generated_ids=generated_ids,
        )

    @contextmanager
    def capture_vision_and_projector(self, model):
        store = {"vision": None, "projector": None}
        handles = []
        vt = getattr(model, "vision_tower", None)
        mmp = getattr(model, "multi_modal_projector", None)
        if vt is not None:
            handles.append(vt.register_forward_hook(
                lambda m, i, o: store.__setitem__("vision", o)))
        if mmp is not None:
            handles.append(mmp.register_forward_hook(
                lambda m, i, o: store.__setitem__("projector", o)))
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    def _pool(self, x):
        if x is None:
            return None
        if isinstance(x, (tuple, list)):
            x = x[0]
        if hasattr(x, "last_hidden_state"):
            x = x.last_hidden_state
        if not torch.is_tensor(x):
            return None
        if x.dim() == 3:
            x = x.mean(dim=1).squeeze(0)
        elif x.dim() == 2:
            x = x.mean(dim=0)
        return x.detach().cpu().to(torch.float32)

    def pool_vision_features(self, vision_output):
        return self._pool(vision_output)

    def pool_projector_features(self, projector_output):
        return self._pool(projector_output)
