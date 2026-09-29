"""Machine-local path resolution.

Single source of truth for every machine-specific path in this repo. Values
come from (in priority order):

  1. an environment variable  (CU_OUTPUTS_ROOT, CU_MANIFEST_PATH, ...)
  2. configs/paths.local.yaml  (gitignored — your private copy)
  3. configs/paths.yaml        (committed, ships EMPTY)

Nothing here has a default. If a path is missing you get a loud error naming
the key, the env var, what the path should point to, and where the canonical
copy lives on the lab server — rather than a FileNotFoundError 200 lines deep.

Usage:
    from _paths import get_path, outputs_root

    BASE_OUTPUT = outputs_root()                 # raises if unset
    manifest = get_path("manifest_path")
    semunc = get_path("semantic_uncertainty_utils")
"""

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
SHARED_CONFIG = CONFIG_DIR / "paths.yaml"
LOCAL_CONFIG = CONFIG_DIR / "paths.local.yaml"

# key -> (env var, one-line description, canonical location on the lab server)
PATH_KEYS = {
    "outputs_root": (
        "CU_OUTPUTS_ROOT",
        "Root of extraction outputs; holds <model>_internal_diag_*/ dirs plus "
        "phase_1_2/, phase_3/, phase_4/. ~7 GB for all four models.",
        "/data/kalashkala/continuous-uncertainty/outputs",
    ),
    "manifest_path": (
        "CU_MANIFEST_PATH",
        "Sequence manifest CSV (video_path, question, category, "
        "max_ambiguity_index / center_frames, gt_answer).",
        "/home/kalashkala/acl_rebuttal_form/list_questions_revised.csv",
    ),
    "image_root": (
        "CU_IMAGE_ROOT",
        "Frame root, laid out as <image_root>/<sequence_id>/frame_XXXX.png. "
        "Needed for extraction only.",
        "/home/kalashkala/acl_rebuttal_form/images",
    ),
    "semantic_uncertainty_utils": (
        "CU_SEMUNC_UTILS",
        "Directory containing eval_utils.py from the semantic_uncertainty repo "
        "(https://github.com/jlko/semantic_uncertainty), at "
        "<clone>/semantic_uncertainty/uncertainty/utils. Phase 4 only.",
        "/home/kalashkala/semantic_uncertainty/semantic_uncertainty/uncertainty/utils",
    ),
}

_cache = None


def _load_file_values():
    global _cache
    if _cache is not None:
        return _cache

    values = {}
    for path in (SHARED_CONFIG, LOCAL_CONFIG):  # local overrides shared
        if not path.exists():
            continue
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        for key, value in (loaded.get("paths") or {}).items():
            if value:
                values[key] = str(value)

    _cache = values
    return values


def _missing_path_error(key):
    env_var, description, canonical = PATH_KEYS[key]
    return RuntimeError(
        f"\n"
        f"─────────────────────────────────────────────────────────────────\n"
        f"  Required path '{key}' is not set.\n"
        f"─────────────────────────────────────────────────────────────────\n"
        f"  What it is:\n"
        f"    {description}\n\n"
        f"  Set it in ONE of these ways:\n"
        f"    1. export {env_var}=/your/path\n"
        f"    2. edit  {LOCAL_CONFIG}   (gitignored; copy paths.yaml to it)\n"
        f"    3. edit  {SHARED_CONFIG}  (committed — ships empty on purpose)\n\n"
        f"  Canonical copy on the lab server (rsync it over rather than\n"
        f"  regenerating — see README 'Getting the data'):\n"
        f"    {canonical}\n"
        f"─────────────────────────────────────────────────────────────────\n"
    )


def get_path(key, required=True, must_exist=True):
    """Resolve one path. Raises a descriptive error if unset or missing."""
    if key not in PATH_KEYS:
        raise KeyError(f"Unknown path key '{key}'. Known: {sorted(PATH_KEYS)}")

    env_var = PATH_KEYS[key][0]
    value = os.environ.get(env_var) or _load_file_values().get(key)

    if not value:
        if required:
            raise _missing_path_error(key)
        return None

    resolved = Path(value).expanduser()
    if must_exist and not resolved.exists():
        _, description, canonical = PATH_KEYS[key]
        raise FileNotFoundError(
            f"\n  Path '{key}' is set to:\n    {resolved}\n"
            f"  ...but that does not exist.\n\n"
            f"  What it should be:\n    {description}\n"
            f"  Canonical copy on the lab server:\n    {canonical}\n"
        )
    return resolved


def outputs_root(must_exist=True):
    """Convenience accessor for the path every analysis phase needs."""
    return get_path("outputs_root", must_exist=must_exist)


def describe_all():
    """Print the resolution status of every key. Used by `--check-paths`."""
    values = _load_file_values()
    lines = ["Path configuration:"]
    for key, (env_var, description, canonical) in PATH_KEYS.items():
        from_env = os.environ.get(env_var)
        value = from_env or values.get(key)
        if value:
            source = f"env:{env_var}" if from_env else "paths.yaml"
            exists = "ok" if Path(value).expanduser().exists() else "MISSING ON DISK"
            lines.append(f"  [{exists:>15}] {key:<28} = {value}   ({source})")
        else:
            lines.append(f"  [{'NOT SET':>15}] {key:<28} = (empty)")
            lines.append(f"                    set {env_var}, or fill configs/paths.yaml")
            lines.append(f"                    lab server copy: {canonical}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_all())
