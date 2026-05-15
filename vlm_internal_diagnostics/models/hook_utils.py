"""Forward hooks for grabbing vision encoder and projector outputs."""
from contextlib import contextmanager


@contextmanager
def capture_module_outputs(module):
    """Capture the output tensor of `module` during one forward pass."""
    store = {"output": None}

    def _hook(_mod, _inp, out):
        store["output"] = out

    handle = module.register_forward_hook(_hook)
    try:
        yield store
    finally:
        handle.remove()


@contextmanager
def capture_vision_and_projector(model):
    """Capture both vision_tower output and multi_modal_projector output."""
    store = {"vision": None, "projector": None}

    def _v_hook(_mod, _inp, out):
        store["vision"] = out

    def _p_hook(_mod, _inp, out):
        store["projector"] = out

    h1 = model.vision_tower.register_forward_hook(_v_hook)
    h2 = model.multi_modal_projector.register_forward_hook(_p_hook)
    try:
        yield store
    finally:
        h1.remove()
        h2.remove()
