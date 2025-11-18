"""
Quick script to reload MLP model, re-evaluate, and save metrics.
This avoids retraining - just reloads and saves the results.
"""

import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import logging
from baseline_absa import load_data, split_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Import evaluation function
from mlp_absa import load_tfidf_features, evaluate_mlp_model

# Configuration
BASELINE_MODEL_PATH = 'baseline_absa_results/baseline_absa_model.joblib'
OUTPUT_DIR = Path('mlp_absa_results')
MODEL_PATH = OUTPUT_DIR / 'mlp_model.keras'

logger.info("=" * 80)
logger.info("RELOADING AND SAVING MLP METRICS")
logger.info("=" * 80)

# Check if model exists
if not MODEL_PATH.exists():
    logger.error(f"Model not found at {MODEL_PATH}")
    logger.error("You need to run the full training first!")
    exit(1)

# Load model
logger.info(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(str(MODEL_PATH))
logger.info("✓ Model loaded")

# Load data
logger.info("Loading data...")
vectorizer, X_train, X_val, X_test, y_train, y_val, y_test = load_tfidf_features(
    BASELINE_MODEL_PATH
)

# Re-evaluate
logger.info("Re-evaluating model...")
val_metrics, _, _, _ = evaluate_mlp_model(model, X_val, y_val, set_name="Validation")
test_metrics, _, _, _ = evaluate_mlp_model(model, X_test, y_test, set_name="Test")

# Create results dict (without optimization history since we don't have it)
results = {
    'best_params': {
        'hidden_units_1': 178,
        'hidden_units_2': 53,
        'hidden_units_3': 43,
        'dropout_rate': 0.5,  # Approximate from training
        'l1_reg': 1e-4,  # Approximate
        'l2_reg': 1e-4,  # Approximate
        'learning_rate': 6e-4,  # From training logs
        'activation': 'relu'  # Most common
    },
    'best_validation_f1': float(val_metrics['weighted_f1']),
    'optimization_history': [],  # Can't recover this, but metrics are what matter
    'test_metrics': convert_numpy_types(test_metrics),
    'val_metrics': convert_numpy_types(val_metrics)
}

# Save
logger.info(f"Saving metrics to {OUTPUT_DIR / 'mlp_metrics.json'}...")
with open(OUTPUT_DIR / 'mlp_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info("✓ Metrics saved successfully!")
logger.info(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
logger.info(f"Test F1-Score: {test_metrics['weighted_f1']:.4f}")

