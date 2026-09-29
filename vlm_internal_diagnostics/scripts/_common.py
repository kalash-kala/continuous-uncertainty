"""Shared helpers for scripts: config loading, path resolution."""
import sys
import os
from pathlib import Path
import yaml

# Make the project importable when scripts are run directly.
PKG_ROOT = Path(__file__).resolve().parent.parent
PROJECT_PARENT = PKG_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Canonical run_name per model family — the directory names the analysis
# phases look up (see MODELS in phase_1_2_analysis.py).
CANONICAL_RUN_NAMES = {
    "llava":           "llava_internal_diag_v1_question_mean_new",
    "phi4_multimodal": "phi4_internal_diag_v1_question_mean_new",
    "pixtral":         "pixtral_internal_diag_v1_question_mean_new",
    "qwen2_5_vl":      "qwen2_5_vl_internal_diag_v1_question_mean_new",
}


def _check_run_name_matches_model(cfgs):
    """Refuse to write one model's outputs into another model's directory.

    Model and run name are set in two different files, so it is easy to switch
    the model preset and forget the run_name — which silently overwrites the
    other model's extraction. A custom run_name is fine; only a run_name that
    is the *canonical name of a different model* is rejected.
    """
    family = cfgs.get("model", {}).get("family")
    run_name = cfgs.get("run", {}).get("run_name")
    if not family or not run_name:
        return

    owner = next((f for f, n in CANONICAL_RUN_NAMES.items() if n == run_name), None)
    if owner is not None and owner != family:
        expected = CANONICAL_RUN_NAMES.get(family, f"<your own name for {family}>")
        raise ValueError(
            f"\n  Model/run mismatch — this would overwrite another model's outputs.\n\n"
            f"    model_config family : {family}\n"
            f"    run_config run_name : {run_name}   <- this is {owner}'s directory\n\n"
            f"  Set `run_name` in run_config.yaml to:\n"
            f"    {expected}\n\n"
            f"  (Any name that is not another model's canonical name is accepted,\n"
            f"   but the analysis phases only look up the canonical ones.)\n"
        )


def _fill_machine_paths(cfgs):
    """Backfill machine-specific paths left empty in the YAML configs.

    Precedence for each value: whatever is written in the config file wins; if
    it is empty we fall back to configs/paths.yaml / the CU_* env vars. This
    lets a fork configure paths in exactly one place (paths.yaml) and leave the
    committed per-run configs untouched.
    """
    from _paths import get_path

    _check_run_name_matches_model(cfgs)

    data = cfgs.get("data", {}).get("data")
    if data is not None:
        if not data.get("manifest_path"):
            data["manifest_path"] = str(get_path("manifest_path"))
        if not data.get("image_root"):
            data["image_root"] = str(get_path("image_root"))

    run = cfgs.get("run")
    if run is not None and not run.get("output_dir"):
        run_name = run.get("run_name")
        if not run_name:
            raise ValueError(
                "run_config.yaml must set either `output_dir` or `run_name`. "
                "With only `run_name`, outputs go to "
                "<outputs_root>/<run_name> using configs/paths.yaml."
            )
        run["output_dir"] = str(get_path("outputs_root", must_exist=False) / run_name)

    return cfgs


def load_all_configs(args):
    """Args namespace from argparse with --*_config paths."""
    cfgs = {}
    if getattr(args, "model_config", None):
        cfgs["model"] = load_yaml(args.model_config)["model"]
        cfgs["model"]["prompt"] = load_yaml(args.model_config).get("prompt", {})
        cfgs["model"]["generation"] = load_yaml(args.model_config).get("generation", {})
    if getattr(args, "data_config", None):
        cfgs["data"] = load_yaml(args.data_config)
    if getattr(args, "extraction_config", None):
        cfgs["extraction"] = load_yaml(args.extraction_config)
    if getattr(args, "visualization_config", None):
        cfgs["visualization"] = load_yaml(args.visualization_config)
    if getattr(args, "run_config", None):
        cfgs["run"] = load_yaml(args.run_config)["run"]
        cfgs["debug"] = load_yaml(args.run_config).get("debug", {})
    return _fill_machine_paths(cfgs)


def resolve_output_dir(run_cfg):
    out = run_cfg.get("output_dir", f"outputs/{run_cfg.get('run_name', 'run')}")
    if not os.path.isabs(out):
        out = str(PKG_ROOT / out)
    return out


def read_jsonl(path):
    import json
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
