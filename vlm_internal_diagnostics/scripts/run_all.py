"""Run extract -> metrics -> visualize in sequence.

Example nohup command:
nohup python -u scripts/run_all.py --model_config configs/model_config.yaml --data_config configs/data_config.yaml --extraction_config configs/extraction_config.yaml --visualization_config configs/visualization_config.yaml --run_config configs/run_config.yaml > run.log 2>&1 &

"""
import argparse
import subprocess
import sys
from pathlib import Path

from _common import load_all_configs, resolve_output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--extraction_config", required=True)
    p.add_argument("--visualization_config", required=True)
    p.add_argument("--run_config", required=True)
    args = p.parse_args()

    cfgs = load_all_configs(args)
    out_dir = resolve_output_dir(cfgs["run"])
    scripts_dir = Path(__file__).parent
    python = sys.executable

    def run(cmd):
        print(f"[run_all] $ {' '.join(cmd)}")
        subprocess.check_call(cmd)

    run([python, str(scripts_dir / "run_extract.py"),
         "--model_config", args.model_config,
         "--data_config", args.data_config,
         "--extraction_config", args.extraction_config,
         "--run_config", args.run_config])

    run([python, str(scripts_dir / "run_metrics.py"),
         "--run_dir", out_dir])

    run([python, str(scripts_dir / "run_visualize.py"),
         "--run_dir", out_dir,
         "--visualization_config", args.visualization_config])

    print(f"[run_all] done. outputs in {out_dir}")


if __name__ == "__main__":
    main()
