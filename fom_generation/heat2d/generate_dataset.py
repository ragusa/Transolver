"""Command-line entry point for heat2d FOM dataset generation."""

import argparse

from .config import apply_overrides, load_config
from .dataset import generate_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="Generate 2D heat-conduction FOM data.")
    parser.add_argument("--config", default="fom_generation/heat2d/config_default.json")
    parser.add_argument("--out", default="fom_generation/data/heat2d_fom_demo")
    parser.add_argument("--n-geometries", type=int, default=None)
    parser.add_argument("--n-params-per-geometry", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mesh-size", type=float, default=None)
    parser.add_argument("--plot-first", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args)
    sample_paths = generate_dataset(config, args.out, plot_first=args.plot_first)
    print(f"Wrote {len(sample_paths)} samples to {args.out}")


if __name__ == "__main__":
    main()
