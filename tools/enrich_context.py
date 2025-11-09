#!/usr/bin/env python3
"""
Context Enrichment Script

Adds realistic variation to environmental context (temp, hour, aqi) in the dataset
to create more diverse training data.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple
import sys


def enrich_context(
    df: pd.DataFrame,
    temp_range: Tuple[float, float] = (18.0, 28.0),
    humidity_range: Tuple[float, float] = (30.0, 70.0),
    aqi_range: Tuple[int, int] = (20, 150),
    hour_distribution: str = "uniform"
) -> pd.DataFrame:
    """
    Enrich dataset with varied environmental context.
    
    Args:
        df: Input DataFrame
        temp_range: Temperature range in Celsius
        humidity_range: Humidity range in percentage
        aqi_range: AQI range
        hour_distribution: 'uniform' or 'realistic' (weighted toward daytime)
    
    Returns:
        Enriched DataFrame
    """
    df = df.copy()
    n = len(df)
    
    # Generate varied temperature (normal distribution around 22°C)
    temp_mean = (temp_range[0] + temp_range[1]) / 2
    temp_std = (temp_range[1] - temp_range[0]) / 6
    df['temp_c'] = np.clip(
        np.random.normal(temp_mean, temp_std, n),
        temp_range[0],
        temp_range[1]
    ).round(1)
    
    # Generate varied humidity (normal distribution around 50%)
    humidity_mean = (humidity_range[0] + humidity_range[1]) / 2
    humidity_std = (humidity_range[1] - humidity_range[0]) / 6
    df['humidity'] = np.clip(
        np.random.normal(humidity_mean, humidity_std, n),
        humidity_range[0],
        humidity_range[1]
    ).round(1)
    
    # Generate varied AQI (skewed toward lower values)
    # Use log-normal distribution for realistic AQI distribution
    aqi_log_mean = np.log((aqi_range[0] + aqi_range[1]) / 2)
    aqi_log_std = 0.5
    df['aqi'] = np.clip(
        np.random.lognormal(aqi_log_mean, aqi_log_std, n),
        aqi_range[0],
        aqi_range[1]
    ).astype(int)
    
    # Generate varied local_hour
    if hour_distribution == "realistic":
        # Weighted toward daytime hours (6-22)
        # Create weights: higher for daytime, lower for nighttime
        hours = np.arange(24)
        weights = np.ones(24)
        # Daytime (6-22): weight 2.0
        weights[6:22] = 2.0
        # Nighttime (22-6): weight 0.5
        weights[22:] = 0.5
        weights[:6] = 0.5
        weights = weights / weights.sum()
        
        df['local_hour'] = np.random.choice(hours, size=n, p=weights)
    else:
        # Uniform distribution
        df['local_hour'] = np.random.randint(0, 24, n)
    
    return df


def main():
    """Main enrichment function."""
    parser = argparse.ArgumentParser(description="Enrich OEM dataset with varied environmental context")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/oem_sample_windows.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/oem_sample_windows_enriched.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--temp-min",
        type=float,
        default=18.0,
        help="Minimum temperature in Celsius"
    )
    parser.add_argument(
        "--temp-max",
        type=float,
        default=28.0,
        help="Maximum temperature in Celsius"
    )
    parser.add_argument(
        "--humidity-min",
        type=float,
        default=30.0,
        help="Minimum humidity percentage"
    )
    parser.add_argument(
        "--humidity-max",
        type=float,
        default=70.0,
        help="Maximum humidity percentage"
    )
    parser.add_argument(
        "--aqi-min",
        type=int,
        default=20,
        help="Minimum AQI"
    )
    parser.add_argument(
        "--aqi-max",
        type=int,
        default=150,
        help="Maximum AQI"
    )
    parser.add_argument(
        "--hour-dist",
        type=str,
        choices=["uniform", "realistic"],
        default="realistic",
        help="Hour distribution: uniform or realistic (weighted toward daytime)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Context Enrichment Script")
    print("=" * 60)
    print()
    
    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records")
    print()
    
    # Enrich context
    print("Enriching environmental context...")
    print(f"  Temperature range: {args.temp_min} - {args.temp_max}°C")
    print(f"  Humidity range: {args.humidity_min} - {args.humidity_max}%")
    print(f"  AQI range: {args.aqi_min} - {args.aqi_max}")
    print(f"  Hour distribution: {args.hour_dist}")
    print()
    
    df_enriched = enrich_context(
        df,
        temp_range=(args.temp_min, args.temp_max),
        humidity_range=(args.humidity_min, args.humidity_max),
        aqi_range=(args.aqi_min, args.aqi_max),
        hour_distribution=args.hour_dist
    )
    
    # Save enriched dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving enriched dataset: {output_path}")
    df_enriched.to_csv(output_path, index=False)
    
    # Print statistics
    print()
    print("Enrichment Statistics:")
    print(f"  Temperature: {df_enriched['temp_c'].min():.1f} - {df_enriched['temp_c'].max():.1f}°C (mean: {df_enriched['temp_c'].mean():.1f}°C)")
    print(f"  Humidity: {df_enriched['humidity'].min():.1f} - {df_enriched['humidity'].max():.1f}% (mean: {df_enriched['humidity'].mean():.1f}%)")
    print(f"  AQI: {df_enriched['aqi'].min()} - {df_enriched['aqi'].max()} (mean: {df_enriched['aqi'].mean():.1f})")
    print(f"  Local hour: {df_enriched['local_hour'].min()} - {df_enriched['local_hour'].max()} (distribution: {dict(df_enriched['local_hour'].value_counts().sort_index())})")
    print()
    print("=" * 60)
    print("[SUCCESS] Context enrichment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

