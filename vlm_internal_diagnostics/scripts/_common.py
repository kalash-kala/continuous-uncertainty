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
    return cfgs


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
