#!/usr/bin/env python3
"""Quick verification script to check EDON CAV setup."""

import sys
import os

def check_imports():
    """Check that all required packages can be imported."""
    print("Checking imports...")
    try:
        import pandas
        import numpy
        import sklearn
        import fastapi
        import uvicorn
        import requests
        import pydantic
        import scipy
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_structure():
    """Check that project structure is correct."""
    print("\nChecking project structure...")
    required_dirs = ['src', 'api', 'tests', 'notebooks', 'docs', 'data', 'models']
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"✗ {dir_name}/ missing")
            all_exist = False
    
    required_files = [
        'cli.py',
        'requirements.txt',
        'README.md',
        'Makefile',
        'Dockerfile',
        'src/pipeline.py',
        'src/features.py',
        'src/embedding.py',
        'src/api_clients.py',
        'api/main.py'
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            all_exist = False
    
    return all_exist

def check_modules():
    """Check that modules can be imported."""
    print("\nChecking module imports...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src import features, embedding, api_clients, pipeline
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Module import error: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 50)
    print("EDON CAV Setup Verification")
    print("=" * 50)
    
    checks = [
        check_imports(),
        check_structure(),
        check_modules()
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("✓ All checks passed! Setup looks good.")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

