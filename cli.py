#!/usr/bin/env python3
"""CLI tool for EDON CAV operations."""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import build_cav_dataset


def main():
    parser = argparse.ArgumentParser(description="EDON CAV CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # build-cav command
    build_parser = subparsers.add_parser("build-cav", help="Build CAV dataset")
    build_parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)"
    )
    build_parser.add_argument(
        "--output",
        type=str,
        default="data/edon_cav.json",
        help="Output path for JSON file (default: data/edon_cav.json)"
    )
    build_parser.add_argument(
        "--lat",
        type=float,
        default=40.7128,
        help="Latitude for environmental data (default: 40.7128)"
    )
    build_parser.add_argument(
        "--lon",
        type=float,
        default=-74.0060,
        help="Longitude for environmental data (default: -74.0060)"
    )
    build_parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for saving models (default: models)"
    )
    
    args = parser.parse_args()
    
    if args.command == "build-cav":
        print(f"Building CAV dataset with {args.n} samples...")
        df = build_cav_dataset(
            n_samples=args.n,
            output_path=args.output,
            lat=args.lat,
            lon=args.lon,
            model_dir=args.model_dir
        )
        print(f"\n✓ Success! Generated {len(df)} records")
        print(f"✓ Saved to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

