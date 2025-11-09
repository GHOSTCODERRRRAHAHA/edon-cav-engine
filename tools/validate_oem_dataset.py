#!/usr/bin/env python3
"""
OEM Dataset Validation Script

Validates OEM dataset files for completeness, data quality, and schema compliance.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def validate_csv(csv_path: Path) -> Tuple[bool, List[str]]:
    """Validate CSV dataset file."""
    errors = []
    warnings = []
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = [
            'window_id', 'window_start_idx', 'cav_raw', 'cav_smooth', 'state',
            'parts_bio', 'parts_env', 'parts_circadian', 'parts_p_stress',
            'temp_c', 'humidity', 'aqi', 'local_hour',
            'eda_mean', 'bvp_std', 'acc_magnitude_mean'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if len(df) == 0:
            errors.append("Dataset is empty")
        else:
            # Check CAV ranges
            if df['cav_raw'].min() < 0 or df['cav_raw'].max() > 10000:
                warnings.append(f"CAV raw out of range [0-10000]: {df['cav_raw'].min()} - {df['cav_raw'].max()}")
            
            if df['cav_smooth'].min() < 0 or df['cav_smooth'].max() > 10000:
                warnings.append(f"CAV smooth out of range [0-10000]: {df['cav_smooth'].min()} - {df['cav_smooth'].max()}")
            
            # Check state values
            valid_states = ['overload', 'balanced', 'focus', 'restorative']
            invalid_states = df[~df['state'].isin(valid_states)]['state'].unique()
            if len(invalid_states) > 0:
                errors.append(f"Invalid state values: {invalid_states.tolist()}")
            
            # Check component parts ranges [0-1]
            for part in ['parts_bio', 'parts_env', 'parts_circadian', 'parts_p_stress']:
                if part in df.columns:
                    if df[part].min() < 0 or df[part].max() > 1:
                        warnings.append(f"{part} out of range [0-1]: {df[part].min():.3f} - {df[part].max():.3f}")
            
            # Check for missing values
            missing_counts = df[required_cols].isnull().sum()
            if missing_counts.sum() > 0:
                warnings.append(f"Missing values found:\n{missing_counts[missing_counts > 0].to_dict()}")
            
            # Check window_id uniqueness
            if df['window_id'].duplicated().any():
                errors.append("Duplicate window_id values found")
            
            # Check window_id continuity
            if not df['window_id'].is_monotonic_increasing:
                warnings.append("window_id is not monotonically increasing")
    
    except Exception as e:
        errors.append(f"Error reading CSV: {str(e)}")
    
    return len(errors) == 0, errors + warnings


def validate_parquet(parquet_path: Path) -> Tuple[bool, List[str]]:
    """Validate Parquet dataset file."""
    errors = []
    warnings = []
    
    try:
        df = pd.read_parquet(parquet_path)
        
        # Check required columns
        required_cols = [
            'window_id', 'window_start_idx', 'cav_raw', 'cav_smooth', 'state',
            'parts_bio', 'parts_env', 'parts_circadian', 'parts_p_stress',
            'temp_c', 'humidity', 'aqi', 'local_hour',
            'eda_mean', 'bvp_std', 'acc_magnitude_mean'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if len(df) == 0:
            errors.append("Dataset is empty")
        else:
            # Check CAV ranges
            if df['cav_raw'].min() < 0 or df['cav_raw'].max() > 10000:
                warnings.append(f"CAV raw out of range [0-10000]: {df['cav_raw'].min()} - {df['cav_raw'].max()}")
            
            if df['cav_smooth'].min() < 0 or df['cav_smooth'].max() > 10000:
                warnings.append(f"CAV smooth out of range [0-10000]: {df['cav_smooth'].min()} - {df['cav_smooth'].max()}")
            
            # Check state values
            valid_states = ['overload', 'balanced', 'focus', 'restorative']
            invalid_states = df[~df['state'].isin(valid_states)]['state'].unique()
            if len(invalid_states) > 0:
                errors.append(f"Invalid state values: {invalid_states.tolist()}")
            
            # Check component parts ranges [0-1]
            for part in ['parts_bio', 'parts_env', 'parts_circadian', 'parts_p_stress']:
                if part in df.columns:
                    if df[part].min() < 0 or df[part].max() > 1:
                        warnings.append(f"{part} out of range [0-1]: {df[part].min():.3f} - {df[part].max():.3f}")
            
            # Check for missing values
            missing_counts = df[required_cols].isnull().sum()
            if missing_counts.sum() > 0:
                warnings.append(f"Missing values found:\n{missing_counts[missing_counts > 0].to_dict()}")
            
            # Check window_id uniqueness
            if df['window_id'].duplicated().any():
                errors.append("Duplicate window_id values found")
            
            # Check window_id continuity
            if not df['window_id'].is_monotonic_increasing:
                warnings.append("window_id is not monotonically increasing")
    
    except Exception as e:
        errors.append(f"Error reading Parquet: {str(e)}")
    
    return len(errors) == 0, errors + warnings


def validate_jsonl(jsonl_path: Path) -> Tuple[bool, List[str]]:
    """Validate JSONL dataset file."""
    errors = []
    warnings = []
    
    try:
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            errors.append("JSONL file is empty")
            return False, errors
        
        # Validate first few records
        sample_size = min(100, len(lines))
        signal_lengths = {'EDA': 240, 'TEMP': 240, 'BVP': 240, 'ACC_x': 240, 'ACC_y': 240, 'ACC_z': 240}
        
        for i, line in enumerate(lines[:sample_size]):
            try:
                record = json.loads(line)
                
                # Check signal arrays
                for signal_name, expected_length in signal_lengths.items():
                    if signal_name not in record:
                        errors.append(f"Record {i}: Missing signal {signal_name}")
                    elif len(record[signal_name]) != expected_length:
                        errors.append(f"Record {i}: {signal_name} has {len(record[signal_name])} samples, expected {expected_length}")
                
                # Check CSV columns are present
                csv_cols = ['window_id', 'cav_raw', 'cav_smooth', 'state']
                for col in csv_cols:
                    if col not in record:
                        errors.append(f"Record {i}: Missing column {col}")
            
            except json.JSONDecodeError as e:
                errors.append(f"Record {i}: Invalid JSON - {str(e)}")
    
    except Exception as e:
        errors.append(f"Error reading JSONL: {str(e)}")
    
    return len(errors) == 0, errors + warnings


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate OEM dataset files")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/oem_sample_windows.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="outputs/oem_sample_windows.parquet",
        help="Path to Parquet file"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="outputs/oem_sample_windows.jsonl",
        help="Path to JSONL file"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OEM Dataset Validation")
    print("=" * 60)
    print()
    
    all_valid = True
    all_issues = []
    
    # Validate CSV
    csv_path = Path(args.csv)
    if csv_path.exists():
        print(f"Validating CSV: {csv_path}")
        is_valid, issues = validate_csv(csv_path)
        if issues:
            for issue in issues:
                if issue.startswith("Missing") or issue.startswith("Invalid") or issue.startswith("Duplicate") or issue.startswith("Error"):
                    print(f"  [ERROR] {issue}")
                    all_valid = False
                else:
                    print(f"  [WARNING] {issue}")
                    if args.strict:
                        all_valid = False
        else:
            print("  [OK] CSV validation passed")
        print()
    else:
        print(f"  [SKIP] CSV file not found: {csv_path}")
        print()
    
    # Validate Parquet
    parquet_path = Path(args.parquet)
    if parquet_path.exists():
        print(f"Validating Parquet: {parquet_path}")
        is_valid, issues = validate_parquet(parquet_path)
        if issues:
            for issue in issues:
                if issue.startswith("Missing") or issue.startswith("Invalid") or issue.startswith("Duplicate") or issue.startswith("Error"):
                    print(f"  [ERROR] {issue}")
                    all_valid = False
                else:
                    print(f"  [WARNING] {issue}")
                    if args.strict:
                        all_valid = False
        else:
            print("  [OK] Parquet validation passed")
        print()
    else:
        print(f"  [SKIP] Parquet file not found: {parquet_path}")
        print()
    
    # Validate JSONL
    jsonl_path = Path(args.jsonl)
    if jsonl_path.exists():
        print(f"Validating JSONL: {jsonl_path}")
        is_valid, issues = validate_jsonl(jsonl_path)
        if issues:
            for issue in issues:
                if issue.startswith("Missing") or issue.startswith("Invalid") or issue.startswith("Duplicate") or issue.startswith("Error"):
                    print(f"  [ERROR] {issue}")
                    all_valid = False
                else:
                    print(f"  [WARNING] {issue}")
                    if args.strict:
                        all_valid = False
        else:
            print("  [OK] JSONL validation passed")
        print()
    else:
        print(f"  [SKIP] JSONL file not found: {jsonl_path}")
        print()
    
    # Summary
    print("=" * 60)
    if all_valid:
        print("[SUCCESS] All validations passed!")
        return 0
    else:
        print("[FAILURE] Validation failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

