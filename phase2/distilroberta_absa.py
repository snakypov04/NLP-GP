"""
=============================================================================
DistilRoBERTa Fine-tuning for Aspect-Based Sentiment Analysis (ABSA)
=============================================================================

Phase 4: Transformer-based model using DistilRoBERTa
Optimized for Google Colab T4 GPU (16GB VRAM)

Features:
- DistilRoBERTa (faster, smaller than RoBERTa)
- Fine-tuned on ABSA dataset with [TARGET] and [OTHER] tokens
- Mixed precision training for T4 GPU
- Memory-efficient training
- Comprehensive evaluation and comparison with baseline/MLP

Expected Performance: 92-94% accuracy

Author: NLP Engineering Team
Date: 2024
=============================================================================
"""

import sys
import os
import json
import gc
import warnings
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoard, Callback
    )
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    raise ImportError("TensorFlow is required")

# Colab-specific imports
try:
    from IPython.display import display, HTML
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

warnings.filterwarnings('ignore')

# Prevent PyTorch import issues
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Clean up any dummy torch modules
if 'torch' in sys.modules:
    torch_module = sys.modules['torch']
    if not hasattr(torch_module, '__spec__') or torch_module.__spec__ is None:
        del sys.modules['torch']

# Hugging Face imports
try:
    from transformers import (
        TFRobertaForSequenceClassification,
        RobertaTokenizer,
        RobertaConfig
    )
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    error_msg = str(e)
    if 'ncclCommWindowRegister' in error_msg or 'torch' in error_msg.lower():
        print("=" * 80)
        print("PYTORCH/TRANSFORMERS IMPORT ISSUE DETECTED")
        print("=" * 80)
        print("SOLUTION: Restart Python runtime, then run:")
        print("  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("=" * 80)
        raise ImportError(f"PyTorch/transformers import issue: {error_msg}")
    else:
        raise ImportError(f"transformers library import failed: {error_msg}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
MAX_LENGTH = 128  # Longer for ABSA (can handle [TARGET] and [OTHER] tokens)
BATCH_SIZE = 16   # Smaller batch for T4 GPU with DistilRoBERTa
LEARNING_RATE = 2e-5
EPOCHS = 5
PATIENCE = 3
DROPOUT = 0.1

# Paths
OUTPUT_DIR = Path('distilroberta_absa_results')
MODEL_DIR = OUTPUT_DIR / 'models'
PLOTS_DIR = OUTPUT_DIR / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Set random seeds
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# GPU CONFIGURATION
# =============================================================================


def configure_t4_gpu():
    """Configure T4 GPU with mixed precision and memory growth."""
    logger.info("=" * 80)
    logger.info("T4 GPU CONFIGURATION")
    logger.info("=" * 80)

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.warning("No GPU devices found. Using CPU.")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                logger.info(f"✓ GPU detected: {gpu_name}")
            except:
                logger.info(f"✓ GPU detected: {gpu.name}")

        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("✓ Mixed precision (FP16) enabled for T4 GPU")

        return True

    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
        return False

# =============================================================================
# MEMORY MONITORING
# =============================================================================


def print_memory_status(stage: str):
    """Print memory usage status."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / (1024 ** 3)
        logger.info(f"Memory usage ({stage}): {mem_gb:.2f} GB")

# =============================================================================
# DATA LOADING
# =============================================================================


def load_and_prepare_data(
    data_file: str = 'absa_dataset.csv',
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare ABSA data (80/10/10 split).

    Returns:
        Tuple of (train_df, val_df, test_df, y_train, y_val, y_test)
    """
    logger.info("=" * 80)
    logger.info("LOADING AND PREPARING ABSA DATA")
    logger.info("=" * 80)

    logger.info(f"Loading dataset: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"✓ Loaded {len(df):,} samples")

    # Use clean_text (already has [TARGET] and [OTHER] tokens)
    df['clean_text'] = df['clean_text'].fillna('')

    # Handle empty texts
    empty_mask = df['clean_text'].str.strip() == ''
    if empty_mask.sum() > 0:
        logger.warning(
            f"Found {empty_mask.sum()} empty texts - filling with 'unknown'")
        df.loc[empty_mask, 'clean_text'] = 'unknown'

    # Get labels
    y = df['sentiment'].values

    # First split: train+val vs test
    df_temp, test_df, y_temp, y_test = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (train_size + val_size)
    train_df, val_df, y_train, y_val = train_test_split(
        df_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )

    logger.info(f"✓ Training samples: {len(train_df):,}")
    logger.info(f"✓ Validation samples: {len(val_df):,}")
    logger.info(f"✓ Test samples: {len(test_df):,}")
    logger.info(
        f"  Sentiment distribution (train): {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    logger.info(
        f"  Sentiment distribution (val): {pd.Series(y_val).value_counts().sort_index().to_dict()}")
    logger.info(
        f"  Sentiment distribution (test): {pd.Series(y_test).value_counts().sort_index().to_dict()}")

    return train_df, val_df, test_df, y_train, y_val, y_test

# =============================================================================
# DATASET PREPARATION
# =============================================================================


def prepare_distilroberta_dataset(
    texts: pd.Series,
    labels: np.ndarray,
    tokenizer: RobertaTokenizer,
    max_length: int = MAX_LENGTH
) -> tf.data.Dataset:
    """
    Prepare dataset for DistilRoBERTa training.

    Args:
        texts: Series of text strings (with [TARGET] and [OTHER] tokens)
        labels: Array of labels (-1, 0, 1)
        tokenizer: DistilRoBERTa tokenizer
        max_length: Maximum sequence length

    Returns:
        TensorFlow dataset
    """
    # Map labels to 0, 1, 2 for model
    label_map = {-1: 0, 0: 1, 1: 2}
    labels_mapped = np.array([label_map[label] for label in labels])

    # Tokenize (preserves [TARGET] and [OTHER] as tokens)
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        },
        labels_mapped
    ))

    return dataset

# =============================================================================
# DISTILROBERTA MODEL SETUP
# =============================================================================


def load_distilroberta_model(num_labels: int = 3):
    """
    Load DistilRoBERTa model and tokenizer.

    Args:
        num_labels: Number of classification labels (3: negative, neutral, positive)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("=" * 80)
    logger.info("LOADING DISTILROBERTA MODEL")
    logger.info("=" * 80)

    model_name = "distilroberta-base"
    logger.info(f"Loading {model_name}...")

    # Optional: Get token from environment if needed
    token = None
    hf_token = os.environ.get(
        'HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if hf_token:
        token = hf_token
        logger.info("✓ Using Hugging Face token from environment")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name, token=token)
    logger.info("✓ Tokenizer loaded")

    # Add [TARGET] and [OTHER] as special tokens if not already present
    special_tokens = ['[TARGET]', '[OTHER]']
    tokenizer.add_tokens(special_tokens)
    logger.info(f"✓ Added special tokens: {special_tokens}")

    # Load model configuration
    config = RobertaConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
        token=token
    )

    # Load model
    model = TFRobertaForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        token=token
    )

    # Resize token embeddings to include new special tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info("✓ Model loaded and token embeddings resized")

    # Compile model
    try:
        optimizer = tf.keras.optimizers.get({
            'class_name': 'Adam',
            'config': {'learning_rate': LEARNING_RATE}
        })
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )
    except Exception as e1:
        try:
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy']
            )
            model.optimizer.learning_rate.assign(LEARNING_RATE)
            logger.info(
                f"Compiled with 'adam' and set learning rate to {LEARNING_RATE}")
        except Exception as e2:
            logger.error(f"Failed to compile model: {e1}, {e2}")
            raise

    logger.info(f"✓ Model compiled (dropout={DROPOUT}, lr={LEARNING_RATE})")

    return model, tokenizer

# =============================================================================
# CLASS WEIGHTS
# =============================================================================


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights to handle imbalance."""
    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = np.array([label_map[label] for label in y])

    classes = np.unique(y_mapped)
    class_weights_array = compute_class_weight(
        'balanced', classes=classes, y=y_mapped
    )

    class_weights = {}
    for i, cls in enumerate(classes):
        class_weights[cls] = float(class_weights_array[i])

    logger.info(f"Class weights: {class_weights}")
    return class_weights

# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================


class SavedModelCheckpoint(Callback):
    """Custom callback to save model in SavedModel format."""

    def __init__(self, filepath, monitor='val_loss', save_best_only=True, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if self.save_best_only:
            if current < self.best:
                self.best = current
                if os.path.exists(self.filepath):
                    shutil.rmtree(self.filepath)
                self.model.save(self.filepath, save_format='tf')
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch + 1}: {self.monitor} improved to {current:.4f}, model saved")
        else:
            self.model.save(self.filepath, save_format='tf')

# =============================================================================
# TRAINING
# =============================================================================


def train_distilroberta(
    model: TFRobertaForSequenceClassification,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    class_weights: Optional[Dict] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE
) -> tf.keras.callbacks.History:
    """Train DistilRoBERTa model."""
    logger.info("=" * 80)
    logger.info("TRAINING DISTILROBERTA")
    logger.info("=" * 80)

    # Batch datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Prepare callbacks
    callbacks = []

    def patch_callback(cb):
        """Add missing methods to callbacks for transformers compatibility."""
        if not hasattr(cb, '_implements_train_batch_hooks'):
            cb._implements_train_batch_hooks = lambda: False
        if not hasattr(cb, '_implements_test_batch_hooks'):
            cb._implements_test_batch_hooks = lambda: False
        if not hasattr(cb, '_implements_predict_batch_hooks'):
            cb._implements_predict_batch_hooks = lambda: False
        return cb

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(patch_callback(early_stopping))

    # Model checkpointing
    checkpoint_path = MODEL_DIR / 'distilroberta_best_model'
    checkpoint = SavedModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(patch_callback(checkpoint))

    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(patch_callback(reduce_lr))

    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Early stopping patience: {patience}")
    if class_weights:
        logger.info(f"  Class weights: {class_weights}")

    print_memory_status("Before training")

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print_memory_status("After training")
    logger.info("✓ Training completed")

    return history

# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_distilroberta(
    model: TFRobertaForSequenceClassification,
    test_dataset: tf.data.Dataset,
    y_test: np.ndarray,
    tokenizer: RobertaTokenizer
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
    """Evaluate DistilRoBERTa model."""
    logger.info("=" * 80)
    logger.info("EVALUATING DISTILROBERTA")
    logger.info("=" * 80)

    # Prepare test dataset
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Predictions
    logger.info("Making predictions...")
    predictions = model.predict(test_dataset, verbose=1)
    logits = predictions.logits
    y_pred_proba = tf.nn.softmax(logits, axis=-1).numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Map predictions back to original labels (-1, 0, 1)
    label_map_reverse = {0: -1, 1: 0, 2: 1}
    y_pred_mapped = np.array([label_map_reverse[pred] for pred in y_pred])

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_mapped)
    weighted_f1 = f1_score(y_test, y_pred_mapped, average='weighted')
    macro_f1 = f1_score(y_test, y_pred_mapped, average='macro')

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred_mapped, labels=[-1, 0, 1], zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred_mapped, labels=[-1, 0, 1])

    # ROC AUC
    y_test_binarized = label_binarize(y_test, classes=[-1, 0, 1])
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
            roc_auc_scores[f'class_{class_label}'] = float(roc_auc)
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
        'classification_report': classification_report(
            y_test, y_pred_mapped, labels=[-1, 0, 1], output_dict=True
        )
    }

    logger.info(f"✓ Accuracy: {accuracy:.4f}")
    logger.info(f"✓ Weighted F1: {weighted_f1:.4f}")
    logger.info(f"✓ Macro F1: {macro_f1:.4f}")
    logger.info(f"✓ Negative F1: {f1[0]:.4f}")
    logger.info(f"✓ Neutral F1: {f1[1]:.4f}")
    logger.info(f"✓ Positive F1: {f1[2]:.4f}")

    return metrics, y_pred_mapped, proba_ordered, roc_curves

# =============================================================================
# VISUALIZATIONS
# =============================================================================


def plot_training_history(history: tf.keras.callbacks.History, save_path: str) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'],
                 label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'],
                 label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'],
                 label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if COLAB_AVAILABLE:
        plt.show()
    plt.close()
    logger.info(f"✓ Training history saved: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix - DistilRoBERTa',
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if COLAB_AVAILABLE:
        plt.show()
    plt.close()
    logger.info(f"✓ Confusion matrix saved: {save_path}")


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
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{class_label} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest) - DistilRoBERTa',
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if COLAB_AVAILABLE:
        plt.show()
    plt.close()
    logger.info(f"✓ ROC curves saved: {save_path}")

# =============================================================================
# COMPARISON WITH BASELINE AND MLP
# =============================================================================


def compare_with_previous_models(distilroberta_metrics: Dict) -> None:
    """Compare DistilRoBERTa results with baseline and MLP."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH PREVIOUS MODELS")
    logger.info("=" * 80)

    # Load baseline metrics
    baseline_path = Path('baseline_absa_results/baseline_absa_metrics.json')
    mlp_path = Path('mlp_absa_results/mlp_metrics.json')

    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data['test_metrics']

        logger.info("\nBaseline (TF-IDF + LR):")
        logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {baseline_metrics['weighted_f1']:.4f}")
        logger.info(
            f"  Improvement: {(distilroberta_metrics['accuracy'] - baseline_metrics['accuracy'])*100:.2f}%")

    if mlp_path.exists():
        with open(mlp_path, 'r') as f:
            mlp_data = json.load(f)
        mlp_metrics = mlp_data['test_metrics']

        logger.info("\nMLP (Optimized):")
        logger.info(f"  Accuracy: {mlp_metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {mlp_metrics['weighted_f1']:.4f}")
        logger.info(
            f"  Improvement: {(distilroberta_metrics['accuracy'] - mlp_metrics['accuracy'])*100:.2f}%")

    logger.info("\nDistilRoBERTa:")
    logger.info(f"  Accuracy: {distilroberta_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score: {distilroberta_metrics['weighted_f1']:.4f}")
    logger.info(f"  Macro F1: {distilroberta_metrics['macro_f1']:.4f}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Main function to run DistilRoBERTa fine-tuning pipeline."""
    logger.info("=" * 80)
    logger.info("DISTILROBERTA FINE-TUNING FOR ABSA")
    logger.info("=" * 80)

    # Configure GPU
    gpu_available = configure_t4_gpu()

    # Load data
    train_df, val_df, test_df, y_train, y_val, y_test = load_and_prepare_data(
        data_file='absa_dataset.csv'
    )

    # Load DistilRoBERTa
    model, tokenizer = load_distilroberta_model(num_labels=3)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_distilroberta_dataset(
        train_df['clean_text'], y_train, tokenizer
    )
    val_dataset = prepare_distilroberta_dataset(
        val_df['clean_text'], y_val, tokenizer
    )
    test_dataset = prepare_distilroberta_dataset(
        test_df['clean_text'], y_test, tokenizer
    )

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)

    # Train model
    history = train_distilroberta(
        model, train_dataset, val_dataset,
        class_weights=class_weights
    )

    best_model_path = MODEL_DIR / 'distilroberta_best_model'
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(
        "Using model with best weights (already restored by early stopping)")

    # Evaluate
    metrics, y_pred, y_pred_proba, roc_curves = evaluate_distilroberta(
        model, test_dataset, y_test, tokenizer
    )

    # Compare with previous models
    compare_with_previous_models(metrics)

    # Save metrics
    metrics_path = OUTPUT_DIR / 'distilroberta_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved: {metrics_path}")

    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    history_path = OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"✓ Training history saved: {history_path}")

    # Generate visualizations
    logger.info("\n📊 Generating visualizations...")
    plot_training_history(history, save_path=str(
        PLOTS_DIR / 'training_history.png'))
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        save_path=str(PLOTS_DIR / 'confusion_matrix.png')
    )
    plot_roc_curves(
        roc_curves,
        metrics,
        save_path=str(PLOTS_DIR / 'roc_curves.png')
    )

    # Print summary
    logger.info("=" * 80)
    logger.info("DISTILROBERTA TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"✓ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"✓ Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"✓ Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"✓ Model saved to: {best_model_path}")
    logger.info(f"✓ Metrics saved to: {metrics_path}")
    logger.info(f"✓ Plots saved to: {PLOTS_DIR}")
    logger.info("=" * 80)

    return model, metrics, history


if __name__ == "__main__":
    main()
