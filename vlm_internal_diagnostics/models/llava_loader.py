"""Load LLaVA model + processor from HuggingFace."""
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def load_llava(model_cfg):
    """Returns (model, processor) per the model section of model_config.yaml."""
    name = model_cfg["name"]
    dtype = _DTYPE_MAP.get(str(model_cfg.get("torch_dtype", "float16")).lower(), torch.float16)
    attn_impl = model_cfg.get("attn_implementation", "eager")

    processor = AutoProcessor.from_pretrained(name)

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
    # Force eager attention so output_attentions returns tensors.
    try:
        model.config._attn_implementation = "eager"
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = "eager"
    except Exception:
        pass
    model.eval()
    return model, processor


def build_prompt(processor, question, prompt_cfg):
    """Build the input prompt string for a single image+question."""
    if prompt_cfg.get("use_chat_template", True):
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }]
        try:
            return processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception:
            pass
    return prompt_cfg.get("fallback_template",
                          "USER: <image>\n{question}\nASSISTANT:").format(question=question)
