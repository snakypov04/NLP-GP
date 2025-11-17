"""
=============================================================================
BiLSTM Model for Financial Sentiment Analysis
=============================================================================

This script implements a Bidirectional LSTM neural network for financial
sentiment analysis using GloVe embeddings.

Features:
- Bidirectional LSTM architecture
- GloVe 300d embeddings (42B or 6B)
- Early stopping and model checkpointing
- Comprehensive evaluation metrics
- Training history visualization
- Model saving/loading

Optimized for: Google Colab T4 GPU (16GB VRAM)
Author: NLP Engineering Team
Date: 2024
=============================================================================
"""

import os
import json
import gc
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Embedding, Bidirectional, LSTM, Dense, Dropout,
        BatchNormalization, GlobalMaxPooling1D
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoard
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    raise ImportError(
        "TensorFlow is required. Install with: pip install tensorflow")

# Scikit-learn metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Import GloVe embeddings module
from glove_embeddings import (
    prepare_glove_data, configure_gpu, print_memory_status,
    clear_memory, MAX_WORDS, MAX_LENGTH, EMBEDDING_DIM
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42
OUTPUT_DIR = Path('bilstm_results')
MODEL_DIR = OUTPUT_DIR / 'models'
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


def build_bilstm_model(
    embedding_matrix: np.ndarray,
    vocab_size: int,
    embedding_dim: int = EMBEDDING_DIM,
    max_length: int = MAX_LENGTH,
    lstm_units: int = 128,
    dropout_rate: float = 0.5,
    dense_units: int = 64,
    num_classes: int = 3,
    trainable_embeddings: bool = False
) -> tf.keras.Model:
    """
    Build a Bidirectional LSTM model for sentiment classification.

    Args:
        embedding_matrix: Pre-trained GloVe embedding matrix
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension (default: 300)
        max_length: Maximum sequence length (default: 30)
        lstm_units: Number of LSTM units (default: 128)
        dropout_rate: Dropout rate (default: 0.5)
        dense_units: Number of dense layer units (default: 64)
        num_classes: Number of output classes (default: 3 for -1, 0, 1)
        trainable_embeddings: Whether to fine-tune embeddings (default: False)

    Returns:
        Compiled Keras model
    """
    logger.info("=" * 80)
    logger.info("BUILDING BILSTM MODEL")
    logger.info("=" * 80)

    model = Sequential([
        # Embedding layer with GloVe weights
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            weights=[embedding_matrix],
            trainable=trainable_embeddings,
            name='embedding'
        ),

        # First Bidirectional LSTM layer
        Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                name='lstm_1'
            ),
            name='bilstm_1'
        ),
        BatchNormalization(name='bn_1'),

        # Second Bidirectional LSTM layer
        Bidirectional(
            LSTM(
                lstm_units // 2,
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                name='lstm_2'
            ),
            name='bilstm_2'
        ),
        BatchNormalization(name='bn_2'),

        # Dense layers
        Dense(dense_units, activation='relu', name='dense_1'),
        Dropout(dropout_rate, name='dropout_1'),
        BatchNormalization(name='bn_3'),

        Dense(dense_units // 2, activation='relu', name='dense_2'),
        Dropout(dropout_rate, name='dropout_2'),

        # Output layer (3 classes: -1, 0, 1)
        Dense(num_classes, activation='softmax', name='output')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Build the model with input shape to enable parameter counting
    model.build(input_shape=(None, max_length))

    logger.info("✓ Model architecture:")
    model.summary(print_fn=logger.info)

    # Count parameters (now that model is built)
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w)
                           for w in model.trainable_weights])
    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_bilstm_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.1,
    patience: int = 5,
    model_save_path: Optional[str] = None
) -> tf.keras.callbacks.History:
    """
    Train the BiLSTM model with callbacks.

    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        epochs: Maximum number of epochs (default: 50)
        batch_size: Batch size (default: 64)
        validation_split: Validation split (default: 0.1)
        patience: Early stopping patience (default: 5)
        model_save_path: Path to save best model (optional)

    Returns:
        Training history
    """
    logger.info("=" * 80)
    logger.info("TRAINING BILSTM MODEL")
    logger.info("=" * 80)

    # Prepare callbacks
    callbacks = []

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Model checkpointing
    if model_save_path:
        checkpoint = ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)

    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # TensorBoard (optional)
    try:
        tensorboard = TensorBoard(
            log_dir=str(OUTPUT_DIR / 'logs'),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
    except:
        pass

    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max epochs: {epochs}")
    logger.info(f"Early stopping patience: {patience}")

    # Train model
    print_memory_status("Before training")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    print_memory_status("After training")

    logger.info("✓ Training completed")
    return history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
    """
    Evaluate the BiLSTM model and compute various metrics.

    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Test labels

    Returns:
        Tuple of (metrics_dict, y_pred, y_pred_proba, roc_curves)
    """
    logger.info("=" * 80)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 80)

    # Predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Map predictions to original labels (-1, 0, 1)
    label_map = {0: -1, 1: 0, 2: 1}
    y_pred_mapped = np.array([label_map[pred] for pred in y_pred])

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred_mapped)
    weighted_f1 = f1_score(y_test, y_pred_mapped, average='weighted')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred_mapped, labels=[-1, 0, 1], zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_mapped, labels=[-1, 0, 1])

    # ROC AUC (one-vs-rest)
    y_test_binarized = label_binarize(y_test, classes=[-1, 0, 1])
    n_classes = 3

    # Reorder probabilities to match label order (-1, 0, 1)
    proba_ordered = np.zeros_like(y_pred_proba)
    proba_ordered[:, 0] = y_pred_proba[:, 0]  # -1
    proba_ordered[:, 1] = y_pred_proba[:, 1]  # 0
    proba_ordered[:, 2] = y_pred_proba[:, 2]  # 1

    roc_auc_scores = {}
    roc_curves = {}

    for i, class_label in enumerate([-1, 0, 1]):
        if len(np.unique(y_test_binarized[:, i])) > 1:
            fpr, tpr, _ = roc_curve(
                y_test_binarized[:, i], proba_ordered[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[f'class_{class_label}'] = roc_auc
            roc_curves[f'class_{class_label}'] = (fpr, tpr)

    # Compile metrics
    metrics = {
        'accuracy': float(accuracy),
        'weighted_f1': float(weighted_f1),
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
        'classification_report': classification_report(
            y_test, y_pred_mapped, labels=[-1, 0, 1], output_dict=True
        )
    }

    logger.info(f"✓ Accuracy: {accuracy:.4f}")
    logger.info(f"✓ Weighted F1: {weighted_f1:.4f}")
    logger.info(f"✓ Negative F1: {f1[0]:.4f}")
    logger.info(f"✓ Neutral F1: {f1[1]:.4f}")
    logger.info(f"✓ Positive F1: {f1[2]:.4f}")

    return metrics, y_pred_mapped, proba_ordered, roc_curves


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history: tf.keras.callbacks.History, save_path: str) -> None:
    """
    Plot training history (loss and accuracy).

    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'],
                 label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history.history['accuracy'],
                 label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'],
                 label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Plot and save confusion matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix - BiLSTM', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_classification_report_table(metrics: Dict, save_path: str) -> None:
    """Plot classification report as a table."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    data = []
    per_class = metrics['per_class_metrics']

    for label, label_name in [('negative', 'Negative'), ('neutral', 'Neutral'), ('positive', 'Positive')]:
        data.append([
            label_name,
            f"{per_class[label]['precision']:.3f}",
            f"{per_class[label]['recall']:.3f}",
            f"{per_class[label]['f1']:.3f}",
            per_class[label]['support']
        ])

    data.append([
        'Weighted Avg',
        f"{metrics['classification_report']['weighted avg']['precision']:.3f}",
        f"{metrics['classification_report']['weighted avg']['recall']:.3f}",
        f"{metrics['classification_report']['weighted avg']['f1-score']:.3f}",
        metrics['classification_report']['weighted avg']['support']
    ])

    table = ax.table(
        cellText=data,
        colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(data) + 1):
        for j in range(5):
            if i == len(data):
                table[(i, j)].set_facecolor('#E8F5E9')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    plt.title('Classification Report - BiLSTM',
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Classification report saved to {save_path}")


def plot_roc_curves(roc_curves: Dict, metrics: Dict, save_path: str) -> None:
    """Plot ROC curves for each class (one-vs-rest)."""
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
    plt.title('ROC Curves (One-vs-Rest) - BiLSTM',
              fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def map_labels_to_indices(y: np.ndarray) -> np.ndarray:
    """
    Map sentiment labels (-1, 0, 1) to indices (0, 1, 2) for model training.

    Args:
        y: Labels array with values -1, 0, 1

    Returns:
        Mapped labels array with values 0, 1, 2
    """
    label_map = {-1: 0, 0: 1, 1: 2}
    return np.array([label_map[label] for label in y])


def save_metrics(metrics: Dict, save_path: str) -> None:
    """Save metrics to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {save_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run the BiLSTM model pipeline."""
    logger.info("=" * 80)
    logger.info("BILSTM MODEL - FINANCIAL SENTIMENT ANALYSIS")
    logger.info("=" * 80)

    # Configure GPU
    gpu_available = configure_gpu(
        memory_fraction=0.9,
        enable_mixed_precision=True
    )

    # Prepare data with GloVe embeddings
    logger.info("\n📊 Preparing data with GloVe embeddings...")
    X_train, X_test, y_train, y_test, embedding_matrix, tokenizer, metadata = prepare_glove_data(
        data_file='phase1_dataset.csv',
        max_words=MAX_WORDS,
        max_length=MAX_LENGTH,
        test_size=0.2,
        random_state=RANDOM_SEED,
        glove_config='42B',
        use_cached=True
    )

    # Map labels to indices for model training
    y_train_mapped = map_labels_to_indices(y_train)
    y_test_mapped = map_labels_to_indices(y_test)

    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_mapped,
        test_size=0.1,
        stratify=y_train_mapped,
        random_state=RANDOM_SEED
    )

    logger.info(f"Training samples: {len(X_train_split):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Test samples: {len(X_test):,}")

    # Build model
    vocab_size = metadata['vocab_size']
    model = build_bilstm_model(
        embedding_matrix=embedding_matrix,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_LENGTH,
        lstm_units=128,
        dropout_rate=0.5,
        dense_units=64,
        num_classes=3,
        trainable_embeddings=False
    )

    # Train model
    model_path = MODEL_DIR / 'bilstm_best_model.h5'
    history = train_bilstm_model(
        model=model,
        X_train=X_train_split,
        y_train=y_train_split,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=64,
        patience=5,
        model_save_path=str(model_path)
    )

    # Load best model
    if model_path.exists():
        logger.info(f"Loading best model from {model_path}")
        model = load_model(model_path)

    # Evaluate model
    metrics, y_pred, y_pred_proba, roc_curves = evaluate_model(
        model, X_test, y_test
    )

    # Save metrics
    metrics_path = OUTPUT_DIR / 'bilstm_metrics.json'
    save_metrics(metrics, str(metrics_path))

    # Generate visualizations
    logger.info("\n📊 Generating visualizations...")
    plot_training_history(history, str(PLOTS_DIR / 'training_history.png'))
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        str(PLOTS_DIR / 'confusion_matrix.png')
    )
    plot_classification_report_table(
        metrics,
        str(PLOTS_DIR / 'classification_report.png')
    )
    plot_roc_curves(
        roc_curves,
        metrics,
        str(PLOTS_DIR / 'roc_curves.png')
    )

    # Print summary
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"✓ Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(
        f"✓ Negative F1: {metrics['per_class_metrics']['negative']['f1']:.4f}")
    logger.info(
        f"✓ Neutral F1: {metrics['per_class_metrics']['neutral']['f1']:.4f}")
    logger.info(
        f"✓ Positive F1: {metrics['per_class_metrics']['positive']['f1']:.4f}")
    logger.info(f"✓ Model saved to: {model_path}")
    logger.info(f"✓ Metrics saved to: {metrics_path}")
    logger.info(f"✓ Plots saved to: {PLOTS_DIR}")
    logger.info("=" * 80)

    # Clean up
    clear_memory()

    return model, metrics, history


if __name__ == "__main__":
    main()
