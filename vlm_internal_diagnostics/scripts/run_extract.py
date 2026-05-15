"""Run extraction phase: load model + dataset, write per_frame_outputs.jsonl."""
import argparse
from pathlib import Path

from _common import load_all_configs, resolve_output_dir
from vlm_internal_diagnostics.data import SequenceDataset
from vlm_internal_diagnostics.models import load_llava
from vlm_internal_diagnostics.extraction import extract_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--extraction_config", required=True)
    p.add_argument("--run_config", required=True)
    args = p.parse_args()

    cfgs = load_all_configs(args)
    out_dir = resolve_output_dir(cfgs["run"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data_cfg = cfgs["data"]["data"]
    debug = cfgs.get("debug", {}) or {}
    max_seq = debug.get("limit_sequences") if debug.get("enabled") else data_cfg.get("max_sequences")
    limit_frames = debug.get("limit_frames_per_sequence") if debug.get("enabled") else None

    ds = SequenceDataset(
        manifest_path=data_cfg["manifest_path"],
        image_root=data_cfg["image_root"],
        max_sequences=max_seq,
        categories=data_cfg.get("categories"),
        limit_frames_per_sequence=limit_frames,
    )
    sequences = list(ds)
    print(f"[extract] {len(sequences)} sequences loaded")

    model, processor = load_llava(cfgs["model"])
    extract_all(model, processor, sequences,
                {"model": cfgs["model"], "extraction": cfgs["extraction"], "run": cfgs["run"]},
                out_dir)
    print(f"[extract] wrote {out_dir}/per_frame_outputs.jsonl")


if __name__ == "__main__":
    main()
