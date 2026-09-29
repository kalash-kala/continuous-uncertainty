"""Phi-4-multimodal-instruct adapter.

Notes:
  - Uses AutoModelForCausalLM with trust_remote_code=True; the model is a
    composite (image + audio + text).
  - num_crops=1 keeps the image as a single 448x448 crop, which (after the
    image encoder + 2x2 token merging used by Phi-4) yields 1024 image
    tokens in the LM sequence. If the actual count differs at runtime we
    fall back to whatever we observe in input_ids.
  - Prompt template: <|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n
  - The image embedding module is at model.model.embed_tokens_extend.image_embed.
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


@register_adapter("phi4_multimodal")
class Phi4MultimodalAdapter(ModelAdapter):
    family = "phi4_multimodal"
    fixed_input_size = 448
    expected_num_patches = 1024  # 32 x 32 after 2x2 merge with num_crops=1
    expected_grid_shape = (32, 32)

    def load(self, model_cfg):
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
        name = model_cfg["name"]
        dtype = _DTYPE_MAP.get(str(model_cfg.get("torch_dtype", "bfloat16")).lower(),
                               torch.bfloat16)
        attn_impl = "eager"  # required for output_attentions

        processor = AutoProcessor.from_pretrained(
            name, trust_remote_code=True, num_crops=1,
        )

        # Load config first and force eager attention to prevent flash_attn2 auto-enable
        config = AutoConfig.from_pretrained(name, trust_remote_code=True)
        config._attn_implementation = "eager"
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"

        kwargs = dict(
            torch_dtype=dtype,
            device_map=model_cfg.get("device_map", "auto"),
            attn_implementation=attn_impl,
            trust_remote_code=True,
            config=config,
        )
        if model_cfg.get("load_in_4bit"):
            kwargs["load_in_4bit"] = True
        if model_cfg.get("load_in_8bit"):
            kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        try:
            model.config._attn_implementation = "eager"
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"
        except Exception:
            pass
        model.eval()
        return model, processor

    def build_prompt(self, processor, question, prompt_cfg):
        messages = [{"role": "user", "content": f"<|image_1|>\n{question}"}]
        try:
            return processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return prompt_cfg.get(
                "fallback_template",
                "<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n",
            ).format(question=question)

    def _find_image_embed_module(self, model):
        for path in (
            "model.embed_tokens_extend.image_embed",
            "model.model.embed_tokens_extend.image_embed",
        ):
            cur = model
            ok = True
            for part in path.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    ok = False; break
            if ok:
                return cur
        return None

    def get_token_indices(self, processor, model, image, question, prompt, generated_ids=None):
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]

        # Phi-4 processor replaces <|image_1|> with <|endoftext10|> (ID 200010)
        # via regex, then pre-expands that single token into num_img_tokens copies
        # of 200010 in input_ids before returning. Use processor.special_image_token_id
        # if available, else fall back to the known constant.
        img_tok = None
        try:
            img_tok = int(processor.special_image_token_id)
        except Exception:
            pass
        if img_tok is None:
            try:
                img_tok = int(processor.tokenizer.convert_tokens_to_ids("<|endoftext10|>"))
                if img_tok == processor.tokenizer.unk_token_id:
                    img_tok = None
            except Exception:
                pass
        if img_tok is None:
            # Older processor versions may use negative sentinel IDs
            unique_neg = [int(t) for t in input_ids.unique().tolist() if int(t) < 0]
            if unique_neg:
                img_tok = unique_neg[0]

        if img_tok is not None:
            num_img = int((input_ids == img_tok).sum())
            if num_img == 0:
                num_img = self.expected_num_patches
        else:
            num_img = self.expected_num_patches

        return build_token_indices_from_ids(
            processor, input_ids,
            inputs.get("attention_mask"), inputs.get("input_image_embeds", inputs.get("pixel_values")),
            image_token_id=img_tok,
            num_image_tokens=num_img,
            generated_ids=generated_ids,
        )

    @contextmanager
    def capture_vision_and_projector(self, model):
        store = {"vision": None, "projector": None}
        img_embed = self._find_image_embed_module(model)
        handles = []
        if img_embed is not None:
            # image_embed output is the projected per-patch features that go
            # into the LM. We treat that as the "projector" output.
            handles.append(img_embed.register_forward_hook(
                lambda m, i, o: store.__setitem__("projector", o)))
            # Try to hook the inner vision encoder if it exposes one.
            for attr in ("img_processor", "vision_tower", "vision_model"):
                vt = getattr(img_embed, attr, None)
                if vt is not None:
                    handles.append(vt.register_forward_hook(
                        lambda m, i, o, _s=store: _s.__setitem__("vision", o)))
                    break
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
