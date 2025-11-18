"""
=============================================================================
Optimized MLP for Aspect-Based Sentiment Analysis (ABSA)
=============================================================================

This script implements a Multi-Layer Perceptron (MLP) with Bayesian
hyperparameter optimization for ABSA on TF-IDF features.

Features:
- Bayesian hyperparameter optimization (scikit-optimize)
- 3-layer MLP with configurable architecture
- Class weights for handling imbalance
- Dropout and batch normalization for regularization
- Comprehensive evaluation and comparison with baseline

Author: NLP Engineering Team
Date: 2024
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not available. Install with: pip install scikit-optimize")

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =============================================================================
# DATA LOADING (Reuse from baseline)
# =============================================================================

def load_tfidf_features(baseline_model_path: str) -> Tuple:
    """
    Load TF-IDF features and data splits from baseline model.
    
    Args:
        baseline_model_path: Path to baseline model joblib file
        
    Returns:
        Tuple of (vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf, 
                  y_train, y_val, y_test)
    """
    logger.info("=" * 80)
    logger.info("LOADING TF-IDF FEATURES FROM BASELINE")
    logger.info("=" * 80)
    
    # Load baseline model to get vectorizer
    baseline_data = joblib.load(baseline_model_path)
    vectorizer = baseline_data['vectorizer']
    
    # Reload data and recreate splits (same random_state=42)
    from baseline_absa import load_data, split_data
    
    DATA_FILE = 'absa_dataset.csv'
    X, y = load_data(DATA_FILE)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
    )
    
    # Transform to TF-IDF
    X_train_tfidf = vectorizer.transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Convert sparse matrices to dense arrays for TensorFlow
    X_train_tfidf = X_train_tfidf.toarray()
    X_val_tfidf = X_val_tfidf.toarray()
    X_test_tfidf = X_test_tfidf.toarray()
    
    logger.info(f"✓ TF-IDF features loaded")
    logger.info(f"  Training shape: {X_train_tfidf.shape}")
    logger.info(f"  Validation shape: {X_val_tfidf.shape}")
    logger.info(f"  Test shape: {X_test_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test


# =============================================================================
# MLP MODEL BUILDING
# =============================================================================

def build_mlp_model(
    input_dim: int,
    hidden_units_1: int,
    hidden_units_2: int,
    hidden_units_3: int,
    dropout_rate: float,
    l1_reg: float,
    l2_reg: float,
    learning_rate: float,
    activation: str = 'relu'
) -> tf.keras.Model:
    """
    Build a 3-layer MLP model.
    
    Args:
        input_dim: Input dimension (TF-IDF feature size)
        hidden_units_1: Number of units in first hidden layer
        hidden_units_2: Number of units in second hidden layer
        hidden_units_3: Number of units in third hidden layer
        dropout_rate: Dropout rate
        l1_reg: L1 regularization strength
        l2_reg: L2 regularization strength
        learning_rate: Learning rate
        activation: Activation function
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Input layer
        Dense(
            hidden_units_1,
            activation=activation,
            input_shape=(input_dim,),
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name='dense_1'
        ),
        BatchNormalization(name='bn_1'),
        Dropout(dropout_rate, name='dropout_1'),
        
        # Second hidden layer
        Dense(
            hidden_units_2,
            activation=activation,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name='dense_2'
        ),
        BatchNormalization(name='bn_2'),
        Dropout(dropout_rate, name='dropout_2'),
        
        # Third hidden layer
        Dense(
            hidden_units_3,
            activation=activation,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
            name='dense_3'
        ),
        BatchNormalization(name='bn_3'),
        Dropout(dropout_rate, name='dropout_3'),
        
        # Output layer (3 classes: -1, 0, 1)
        Dense(3, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# BAYESIAN OPTIMIZATION
# =============================================================================

def bayesian_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_calls: int = 30,
    n_initial_points: int = 10
) -> Dict:
    """
    Perform Bayesian hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_calls: Number of optimization iterations
        n_initial_points: Number of random initial points
        
    Returns:
        Dictionary with best hyperparameters and optimization results
    """
    logger.info("\n" + "=" * 80)
    logger.info("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Optimization iterations: {n_calls}")
    logger.info(f"Initial random points: {n_initial_points}")
    
    input_dim = X_train.shape[1]
    
    # Define search space
    dimensions = [
        Integer(128, 512, name='hidden_units_1'),
        Integer(64, 256, name='hidden_units_2'),
        Integer(32, 128, name='hidden_units_3'),
        Real(0.1, 0.5, name='dropout_rate'),
        Real(1e-6, 1e-3, prior='log-uniform', name='l1_reg'),
        Real(1e-6, 1e-3, prior='log-uniform', name='l2_reg'),
        Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
        Categorical(['relu', 'tanh', 'elu'], name='activation')
    ]
    
    # Calculate class weights
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        'balanced', classes=classes, y=y_train
    )
    class_weights = {i: weight for i, weight in enumerate(classes)}
    for i, cls in enumerate(classes):
        class_weights[cls] = class_weights_array[i]
    
    # Track best score
    best_score = -np.inf
    best_params = None
    optimization_history = []
    iteration_count = [0]  # Use list to allow modification in nested function
    
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        """Objective function for optimization."""
        nonlocal best_score, best_params
        iteration_count[0] += 1
        current_iter = iteration_count[0]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimization Iteration {current_iter}/{n_calls}")
        logger.info(f"{'='*60}")
        logger.info(f"Testing parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        try:
            # Build model
            model = build_mlp_model(
                input_dim=input_dim,
                hidden_units_1=params['hidden_units_1'],
                hidden_units_2=params['hidden_units_2'],
                hidden_units_3=params['hidden_units_3'],
                dropout_rate=params['dropout_rate'],
                l1_reg=params['l1_reg'],
                l2_reg=params['l2_reg'],
                learning_rate=params['learning_rate'],
                activation=params['activation']
            )
            
            # Train with early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
            
            logger.info(f"\nTraining model (iteration {current_iter})...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=128,
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weights,
                verbose=1  # Show epoch-by-epoch progress
            )
            
            # Evaluate on validation set
            y_pred = model.predict(X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Map predictions back to original labels (-1, 0, 1)
            label_map = {0: -1, 1: 0, 2: 1}
            y_pred_mapped = np.array([label_map[pred] for pred in y_pred_classes])
            
            # Calculate F1 score (we want to maximize this)
            f1 = f1_score(y_val, y_pred_mapped, average='weighted')
            
            # Store history
            optimization_history.append({
                'params': params.copy(),
                'f1_score': float(f1),
                'epochs': len(history.history['loss'])
            })
            
            # Update best
            logger.info(f"\nIteration {current_iter} Results:")
            logger.info(f"  Validation F1-Score: {f1:.4f}")
            logger.info(f"  Epochs trained: {len(history.history['loss'])}")
            
            if f1 > best_score:
                best_score = f1
                best_params = params.copy()
                logger.info(f"  ⭐ NEW BEST! F1: {f1:.4f}")
            else:
                logger.info(f"  Current best F1: {best_score:.4f}")
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            # Return negative F1 (minimize = maximize F1)
            return -f1
            
        except Exception as e:
            logger.warning(f"Error in optimization: {e}")
            return 1.0  # Return bad score
    
    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=RANDOM_SEED,
        acq_func='EI'  # Expected Improvement
    )
    
    # Get best parameters
    best_params = {
        'hidden_units_1': result.x[0],
        'hidden_units_2': result.x[1],
        'hidden_units_3': result.x[2],
        'dropout_rate': result.x[3],
        'l1_reg': result.x[4],
        'l2_reg': result.x[5],
        'learning_rate': result.x[6],
        'activation': result.x[7]
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best validation F1-score: {-result.fun:.4f}")
    logger.info(f"Best parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    
    return {
        'best_params': best_params,
        'best_score': -result.fun,
        'optimization_history': optimization_history
    }


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict,
    epochs: int = 100,
    batch_size: int = 128
) -> Tuple[tf.keras.Model, Dict]:
    """
    Train MLP model with best hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        best_params: Best hyperparameters from optimization
        epochs: Maximum number of epochs
        batch_size: Batch size
        
    Returns:
        Tuple of (trained model, training history)
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MLP MODEL")
    logger.info("=" * 80)
    
    input_dim = X_train.shape[1]
    
    # Calculate class weights
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        'balanced', classes=classes, y=y_train
    )
    class_weights = {}
    for i, cls in enumerate(classes):
        class_weights[cls] = class_weights_array[i]
    
    # Build model with best parameters
    model = build_mlp_model(
        input_dim=input_dim,
        hidden_units_1=best_params['hidden_units_1'],
        hidden_units_2=best_params['hidden_units_2'],
        hidden_units_3=best_params['hidden_units_3'],
        dropout_rate=best_params['dropout_rate'],
        l1_reg=best_params['l1_reg'],
        l2_reg=best_params['l2_reg'],
        learning_rate=best_params['learning_rate'],
        activation=best_params['activation']
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    
    logger.info("✓ Training completed")
    
    return model, history.history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_mlp_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    set_name: str = "Test"
) -> Dict:
    """
    Evaluate MLP model.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        set_name: Name of the dataset
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info(f"\nEvaluating on {set_name} set...")
    
    # Predictions
    y_pred_proba = model.predict(X, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    # Map predictions back to original labels (-1, 0, 1)
    label_map = {0: -1, 1: 0, 2: 1}
    y_pred = np.array([label_map[pred] for pred in y_pred_classes])
    
    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    macro_f1 = f1_score(y, y_pred, average='macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=[-1, 0, 1], zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=[-1, 0, 1])
    
    # ROC AUC (one-vs-rest)
    y_binarized = label_binarize(y, classes=[-1, 0, 1])
    
    roc_auc_scores = {}
    roc_curves = {}
    
    for i, class_label in enumerate([-1, 0, 1]):
        if len(np.unique(y_binarized[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[f'class_{class_label}'] = roc_auc
            roc_curves[f'class_{class_label}'] = (fpr, tpr)
    
    # Compile metrics
    metrics = {
        'accuracy': float(accuracy),
        'weighted_f1': float(weighted_f1),
        'macro_f1': float(macro_f1),
        'per_class_metrics': {
            'negative': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0])
            },
            'neutral': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1])
            },
            'positive': {
                'precision': float(precision[2]),
                'recall': float(recall[2]),
                'f1': float(f1[2]),
                'support': int(support[2])
            }
        },
        'confusion_matrix': cm.tolist(),
        'roc_auc_scores': roc_auc_scores,
        'classification_report': classification_report(y, y_pred, labels=[-1, 0, 1], output_dict=True)
    }
    
    return metrics, y_pred, y_pred_proba, roc_curves


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_training_history(history: Dict, save_path: str) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Training history saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix - Optimized MLP', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Confusion matrix saved to {save_path}")


def plot_roc_curves(roc_curves: Dict, metrics: Dict, save_path: str) -> None:
    """Plot ROC curves."""
    plt.figure(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green']
    class_labels = ['Negative', 'Neutral', 'Positive']
    class_values = [-1, 0, 1]
    
    for i, (class_val, class_label) in enumerate(zip(class_values, class_labels)):
        if f'class_{class_val}' in roc_curves:
            fpr, tpr = roc_curves[f'class_{class_val}']
            auc_score = metrics['roc_auc_scores'].get(f'class_{class_val}', 0)
            plt.plot(
                fpr, tpr, 
                color=colors[i], 
                lw=2, 
                label=f'{class_label} (AUC = {auc_score:.3f})'
            )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest) - Optimized MLP', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ ROC curves saved to {save_path}")


def plot_optimization_history(optimization_history: List[Dict], save_path: str) -> None:
    """Plot optimization history."""
    if not optimization_history:
        return
    
    f1_scores = [h['f1_score'] for h in optimization_history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, 'o-', linewidth=2, markersize=6)
    plt.axhline(y=max(f1_scores), color='r', linestyle='--', label=f'Best F1: {max(f1_scores):.4f}')
    plt.xlabel('Optimization Iteration', fontsize=12)
    plt.ylabel('Validation F1-Score', fontsize=12)
    plt.title('Bayesian Optimization History', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Optimization history saved to {save_path}")


# =============================================================================
# COMPARISON WITH BASELINE
# =============================================================================

def compare_with_baseline(mlp_metrics: Dict, baseline_metrics_path: str) -> None:
    """Compare MLP results with baseline."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("=" * 80)
    
    # Load baseline metrics
    with open(baseline_metrics_path, 'r') as f:
        baseline_data = json.load(f)
    baseline_metrics = baseline_data['test_metrics']
    
    logger.info("\nAccuracy Comparison:")
    logger.info(f"  Baseline: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"  MLP:      {mlp_metrics['accuracy']:.4f}")
    logger.info(f"  Improvement: {(mlp_metrics['accuracy'] - baseline_metrics['accuracy'])*100:.2f}%")
    
    logger.info("\nWeighted F1-Score Comparison:")
    logger.info(f"  Baseline: {baseline_metrics['weighted_f1']:.4f}")
    logger.info(f"  MLP:      {mlp_metrics['weighted_f1']:.4f}")
    logger.info(f"  Improvement: {(mlp_metrics['weighted_f1'] - baseline_metrics['weighted_f1'])*100:.2f}%")
    
    logger.info("\nPaper's Results (for reference):")
    logger.info(f"  Paper MLP Accuracy: 82.38%")
    logger.info(f"  Paper MLP F1-Score: 75.58%")
    logger.info(f"  Our MLP Accuracy: {mlp_metrics['accuracy']*100:.2f}%")
    logger.info(f"  Our MLP F1-Score: {mlp_metrics['weighted_f1']*100:.2f}%")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run optimized MLP pipeline."""
    logger.info("=" * 80)
    logger.info("OPTIMIZED MLP FOR ABSA")
    logger.info("=" * 80)
    
    if not SKOPT_AVAILABLE:
        logger.error("scikit-optimize is required for Bayesian optimization.")
        logger.error("Install with: pip install scikit-optimize")
        return
    
    # Configuration
    BASELINE_MODEL_PATH = 'baseline_absa_results/baseline_absa_model.joblib'
    BASELINE_METRICS_PATH = 'baseline_absa_results/baseline_absa_metrics.json'
    OUTPUT_DIR = Path('mlp_absa_results')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Load TF-IDF features
    vectorizer, X_train, X_val, X_test, y_train, y_val, y_test = load_tfidf_features(
        BASELINE_MODEL_PATH
    )
    
    # Bayesian optimization
    opt_results = bayesian_optimization(
        X_train, y_train, X_val, y_val,
        n_calls=30,  # Adjust based on time constraints
        n_initial_points=10
    )
    
    # Train final model
    model, history = train_mlp_model(
        X_train, y_train, X_val, y_val,
        opt_results['best_params'],
        epochs=100,
        batch_size=128
    )
    
    # Evaluate
    val_metrics, _, _, _ = evaluate_mlp_model(model, X_val, y_val, set_name="Validation")
    test_metrics, y_pred, y_pred_proba, roc_curves = evaluate_mlp_model(
        model, X_test, y_test, set_name="Test"
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nValidation Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Validation F1-Score: {val_metrics['weighted_f1']:.4f}")
    logger.info(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['weighted_f1']:.4f}")
    
    # Compare with baseline
    compare_with_baseline(test_metrics, BASELINE_METRICS_PATH)
    
    # Visualizations
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    plot_training_history(history, str(PLOTS_DIR / 'training_history.png'))
    plot_confusion_matrix(
        np.array(test_metrics['confusion_matrix']),
        str(PLOTS_DIR / 'confusion_matrix.png')
    )
    plot_roc_curves(
        roc_curves,
        test_metrics,
        str(PLOTS_DIR / 'roc_curves.png')
    )
    plot_optimization_history(
        opt_results['optimization_history'],
        str(PLOTS_DIR / 'optimization_history.png')
    )
    
    # Save results
    logger.info("\nSaving results...")
    model.save(str(OUTPUT_DIR / 'mlp_model'))
    
    results = {
        'best_params': opt_results['best_params'],
        'best_validation_f1': opt_results['best_score'],
        'optimization_history': opt_results['optimization_history'],
        'test_metrics': test_metrics,
        'val_metrics': val_metrics
    }
    
    with open(OUTPUT_DIR / 'mlp_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZED MLP TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    main()

