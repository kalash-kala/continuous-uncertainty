"""Qwen2.5-VL-7B adapter.

Notes:
  - Force min_pixels = max_pixels = 448*448 so vision tower always emits a
    fixed grid. After the patch_merger (2x2 spatial merge) this yields
    32*32 = 1024 image tokens in the LM sequence.
  - MUST use attn_implementation="eager" — the default flash/window kernels do
    not support output_attentions=True and we need attentions.
  - Hooks: vision tower output is captured at model.visual; the post-merger
    features (what actually enters the LM) are the visual module's final
    output. The Qwen2.5-VL "projector" is fused into model.visual.merger.
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


@register_adapter("qwen2_5_vl")
class Qwen25VLAdapter(ModelAdapter):
    family = "qwen2_5_vl"
    fixed_input_size = 448
    expected_num_patches = 256  # 16 x 16 after 2x2 spatial merge (vision tower: 32x32=1024, merger: 16x16=256)
    expected_grid_shape = (16, 16)

    def load(self, model_cfg):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor,
        )
        name = model_cfg["name"]
        dtype = _DTYPE_MAP.get(str(model_cfg.get("torch_dtype", "bfloat16")).lower(),
                               torch.bfloat16)
        # Force eager attention so output_attentions returns tensors.
        attn_impl = "eager"

        px = self.fixed_input_size * self.fixed_input_size
        processor = AutoProcessor.from_pretrained(
            name, min_pixels=px, max_pixels=px,
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

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kwargs)
        try:
            model.config._attn_implementation = "eager"
            if hasattr(model.config, "text_config"):
                model.config.text_config._attn_implementation = "eager"
        except Exception:
            pass
        model.eval()
        return model, processor

    def build_prompt(self, processor, question, prompt_cfg):
        # Qwen2.5-VL chat template handles image placeholders.
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
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n",
            ).format(question=question)

    def get_token_indices(self, processor, model, image, question, prompt, generated_ids=None):
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        # Qwen2.5-VL uses <|image_pad|> as the per-patch image token, already
        # pre-expanded by the processor to the post-merger count.
        img_tok = None
        for attr in ("image_token_index", "image_token_id"):
            v = getattr(model.config, attr, None)
            if v is not None:
                img_tok = int(v); break
        if img_tok is None:
            try:
                img_tok = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            except Exception:
                img_tok = None
        # Count actual image pad tokens from input_ids — the processor pre-expands
        # these to exactly the post-merger LM token count (256 for 448x448), so
        # using the hardcoded expected_num_patches would cause a mismatch.
        actual_num_image_tokens = int((input_ids == img_tok).sum()) if img_tok is not None else self.expected_num_patches
        return build_token_indices_from_ids(
            processor, input_ids,
            inputs.get("attention_mask"), inputs.get("pixel_values"),
            image_token_id=img_tok,
            num_image_tokens=actual_num_image_tokens,
            generated_ids=generated_ids,
        )

    @contextmanager
    def capture_vision_and_projector(self, model):
        store = {"vision": None, "projector": None}
        visual = getattr(model, "visual", None)
        merger = getattr(visual, "merger", None) if visual is not None else None
        handles = []
        if visual is not None:
            handles.append(visual.register_forward_hook(
                lambda m, i, o: store.__setitem__("projector", o)))
        if merger is not None:
            # Pre-merger features (vision tower-like). We capture the input to
            # the merger as the "vision" features since visual already runs the
            # transformer.
            def _pre_merger_hook(_m, inp):
                store["vision"] = inp[0] if isinstance(inp, tuple) and inp else inp
            handles.append(merger.register_forward_pre_hook(_pre_merger_hook))
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    def pool_vision_features(self, vision_output):
        if vision_output is None:
            return None
        feat = vision_output
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat):
            return None
        if feat.dim() == 3:
            feat = feat.mean(dim=1).squeeze(0)
        elif feat.dim() == 2:
            feat = feat.mean(dim=0)
        return feat.detach().cpu().to(torch.float32)

    def pool_projector_features(self, projector_output):
        if projector_output is None:
            return None
        feat = projector_output
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if not torch.is_tensor(feat):
            return None
        if feat.dim() == 3:
            feat = feat.mean(dim=1).squeeze(0)
        elif feat.dim() == 2:
            feat = feat.mean(dim=0)
        return feat.detach().cpu().to(torch.float32)
