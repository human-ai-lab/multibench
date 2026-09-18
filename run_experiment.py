"""Run a MultiBench experiment defined by a YAML config file.

Usage:
    python run_experiment.py --config configs/affect_mosi_late_fusion.yaml
"""
import argparse

from utils.config import run_from_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    args = parser.parse_args()

    results = run_from_file(args.config)
    if results:
        print("Evaluation results:")
        for name, value in results.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
