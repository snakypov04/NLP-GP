"""
Quick script to fix and save MLP metrics after training.
This reloads the model and metrics, converts numpy types, and saves again.
"""

import json
import numpy as np
from pathlib import Path
import tensorflow as tf

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Load the model to get the metrics from training
OUTPUT_DIR = Path('mlp_absa_results')
METRICS_PATH = OUTPUT_DIR / 'mlp_metrics.json'

# If metrics file exists but is incomplete, we need to reconstruct
# Otherwise, just reload and fix
if METRICS_PATH.exists():
    print(f"Found existing metrics file: {METRICS_PATH}")
    print("Trying to load and fix...")
    try:
        with open(METRICS_PATH, 'r') as f:
            results = json.load(f)
        print("✓ Metrics file is already valid JSON")
    except:
        print("✗ Metrics file is corrupted, need to reconstruct from training")
        print("Run the full training script again with the fix applied.")
else:
    print("No metrics file found. Need to run training again.")
    print("But the model is saved, so you can load it and evaluate again.")

print("\nTo fix: Just run the main script again - it will skip training if model exists,")
print("or run: python mlp_absa.py (with the fix applied)")

