#!/usr/bin/env python
"""
End-to-end pipeline runner for XGBoost Fraud Detection Demo.

Usage:
    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --skip-training
    uv run python scripts/run_pipeline.py --api-only
"""

import subprocess
import sys
import time


def run_step(name: str, command: list[str]) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}\n")
    
    start_time = time.time()
    result = subprocess.run(command, cwd=".")
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {name} (exit code {result.returncode})")
        return False
    
    print(f"\n✅ SUCCESS: {name} ({elapsed:.1f}s)")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fraud detection pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only run API")
    parser.add_argument("--api-only", action="store_true", help="Only start the API server")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("XGBoost Fraud Detection Pipeline")
    print("="*60)
    
    if args.api_only:
        print("\nStarting API server only...")
        run_step("Start API Server", [sys.executable, "src/api.py"])
        return
    
    steps = []
    
    if not args.skip_training:
        steps = [
            ("Generate Transaction Data", [sys.executable, "src/data_generator.py"]),
            ("Feature Engineering (PySpark)", [sys.executable, "src/feature_engineering.py"]),
            ("Train XGBoost Model", [sys.executable, "src/train.py"] + (["--gpu"] if args.gpu else [])),
        ]
    
    steps.append(("Start API Server", [sys.executable, "src/api.py"]))
    
    for name, command in steps[:-1]:  # All steps except API
        if not run_step(name, command):
            print("\n❌ Pipeline failed!")
            sys.exit(1)
    
    # Run API server (this blocks)
    name, command = steps[-1]
    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"{'='*60}")
    print("\nAPI will be available at http://localhost:8000")
    print("Press Ctrl+C to stop\n")
    
    subprocess.run(command)


if __name__ == "__main__":
    main()
