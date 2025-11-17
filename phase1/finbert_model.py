"""
=============================================================================
FinBERT Fine-tuning for Financial Sentiment Analysis
=============================================================================

Phase 3: Transformer-based model using ProsusAI/finbert
Optimized for Google Colab T4 GPU (16GB VRAM)

Features:
- Finance-specific BERT (FinBERT) from Hugging Face
- Mixed precision training for T4 GPU
- Memory-efficient training
- Comprehensive evaluation metrics
- Training history visualization

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

# Prevent PyTorch import issues - set environment variables first
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Clean up any dummy torch modules that might interfere
if 'torch' in sys.modules:
    torch_module = sys.modules['torch']
    # Check if it's a dummy module (doesn't have proper __spec__)
    if not hasattr(torch_module, '__spec__') or torch_module.__spec__ is None:
        # Remove the dummy module to allow real torch to import
        del sys.modules['torch']

# Hugging Face imports
# PyTorch should now be properly installed, so we can import transformers normally
try:
    from transformers import (
        TFBertForSequenceClassification,
        BertTokenizer,
        BertConfig
    )
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    error_msg = str(e)
    if 'ncclCommWindowRegister' in error_msg or 'torch' in error_msg.lower() or 'torch.__spec__' in error_msg:
        print("=" * 80)
        print("PYTORCH/TRANSFORMERS IMPORT ISSUE DETECTED")
        print("=" * 80)
        print("The transformers library is having trouble importing PyTorch.")
        print("")
        print("SOLUTION: Try restarting the Python runtime/kernel, then run:")
        print("  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("")
        print("Or if you're in Colab, restart the runtime completely.")
        print("=" * 80)
        raise ImportError(
            "PyTorch/transformers import issue. Please restart the runtime and ensure PyTorch is properly installed.\n"
            f"Original error: {error_msg}"
        )
    else:
        raise ImportError(
            f"transformers library import failed. Error: {error_msg}\n"
            "Install with: pip install transformers"
        )

# Configure logging (must be after imports)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
MAX_LENGTH = 30
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5
PATIENCE = 2
DROPOUT = 0.2

# Paths
OUTPUT_DIR = Path('finbert_results')
MODEL_DIR = OUTPUT_DIR / 'models'
PLOTS_DIR = OUTPUT_DIR / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# TEXT PREPROCESSING (Phase 1 consistency)
# =============================================================================


def clean_text(text: str) -> str:
    """
    Clean text using EXACT Phase 1 preprocessing approach.
    Must match preprocess_dataset.py to ensure consistency.
    """
    if pd.isna(text) or text == '':
        return ''

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    # Remove special characters except apostrophes and spaces
    text = re.sub(r"[^a-z0-9'\s]", '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# =============================================================================
# GPU CONFIGURATION
# =============================================================================

def configure_t4_gpu():
    """Configure T4 GPU with mixed precision and memory growth."""
    logger.info("=" * 80)
    logger.info("T4 GPU CONFIGURATION")
    logger.info("=" * 80)

    # Set random seed
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.warning("No GPU devices found. Using CPU.")
        return False

    try:
        for gpu in gpus:
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpu, True)

            # Get GPU details
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                logger.info(f"✓ GPU detected: {gpu_name}")
            except:
                logger.info(f"✓ GPU detected: {gpu.name}")

        # Enable mixed precision training (FP16)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("✓ Mixed precision (FP16) enabled for T4 GPU")

        # Enable XLA JIT compilation
        tf.config.optimizer.set_jit(True)
        logger.info("✓ XLA JIT compilation enabled")

        return True

    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
        return False


# =============================================================================
# MEMORY MONITORING
# =============================================================================

def get_memory_usage() -> Dict:
    """Get current memory usage statistics."""
    if not PSUTIL_AVAILABLE:
        return {'available': False}

    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'used_gb': mem.used / (1024**3),
        'available_gb': mem.available / (1024**3),
        'percent': mem.percent
    }


def print_memory_status(label: str = ""):
    """Print formatted memory status."""
    if not PSUTIL_AVAILABLE:
        return

    mem = get_memory_usage()
    logger.info(f"Memory Status {f'- {label}' if label else ''}:")
    logger.info(f"  Used: {mem['used_gb']:.2f} GB ({mem['percent']:.1f}%)")
    logger.info(f"  Available: {mem['available_gb']:.2f} GB")

    if mem['percent'] > 90:
        logger.warning("🚨 CRITICAL: Memory usage above 90%!")


class MemoryMonitor(Callback):
    """Callback to monitor memory during training."""

    def on_epoch_end(self, epoch, logs=None):
        if PSUTIL_AVAILABLE:
            mem = get_memory_usage()
            logger.info(f"Epoch {epoch + 1} - Memory: {mem['percent']:.1f}%")

            if mem['percent'] > 85:
                logger.warning("⚠️  High memory usage detected")


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data(
    data_file: str = 'phase1_dataset.csv',
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and prepare data for FinBERT training.

    Returns:
        Tuple of (train_df, test_df, y_train, y_test)
    """
    logger.info("=" * 80)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 80)

    # Load dataset
    logger.info(f"Loading dataset: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"✓ Loaded {len(df):,} samples")

    # Clean texts
    logger.info("Cleaning texts...")
    df['clean_text'] = df['clean_text'].fillna('').apply(clean_text)

    # Handle empty texts
    empty_mask = df['clean_text'].str.strip() == ''
    if empty_mask.sum() > 0:
        logger.warning(
            f"Found {empty_mask.sum()} empty texts - filling with 'unknown'")
        df.loc[empty_mask, 'clean_text'] = 'unknown'

    # Get labels
    y = df['sentiment'].values

    # Train-test split (stratified)
    logger.info(f"Splitting data (test_size={test_size})...")
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(f"✓ Training samples: {len(train_df):,}")
    logger.info(f"✓ Test samples: {len(test_df):,}")
    logger.info(
        f"  Sentiment distribution (train): {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    logger.info(
        f"  Sentiment distribution (test): {pd.Series(y_test).value_counts().sort_index().to_dict()}")

    return train_df, test_df, y_train, y_test


def prepare_finbert_dataset(
    texts: pd.Series,
    labels: np.ndarray,
    tokenizer: BertTokenizer,
    max_length: int = MAX_LENGTH
) -> tf.data.Dataset:
    """
    Prepare dataset for FinBERT training.

    Args:
        texts: Series of text strings
        labels: Array of labels (-1, 0, 1)
        tokenizer: FinBERT tokenizer
        max_length: Maximum sequence length

    Returns:
        TensorFlow dataset
    """
    # Map labels to 0, 1, 2 for model (FinBERT expects 0-indexed)
    label_map = {-1: 0, 0: 1, 1: 2}
    labels_mapped = np.array([label_map[label] for label in labels])

    # Tokenize
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
# FINBERT MODEL SETUP
# =============================================================================

def load_finbert_model(num_labels: int = 3):
    """
    Load FinBERT model and tokenizer.

    Args:
        num_labels: Number of classification labels (3: negative, neutral, positive)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("=" * 80)
    logger.info("LOADING FINBERT MODEL")
    logger.info("=" * 80)

    model_name = "ProsusAI/finbert"
    logger.info(f"Loading {model_name}...")

    # Optional: Get token from environment if needed
    token = None
    hf_token = os.environ.get(
        'HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if hf_token:
        token = hf_token
        logger.info("✓ Using Hugging Face token from environment")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        token=token
    )
    logger.info("✓ Tokenizer loaded")

    # Load model configuration
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
        token=token
    )

    # Load model
    model = TFBertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        token=token
    )
    logger.info("✓ Model loaded")

    # Compile model - transformers models need optimizer as string or use get()
    # The issue is that model.compile() from transformers expects a string identifier
    try:
        # Try using tf.keras.optimizers.get() which accepts string with config
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
        # Fallback: compile with string and set learning rate after
        try:
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy']
            )
            # Set learning rate manually after compilation
            model.optimizer.learning_rate.assign(LEARNING_RATE)
            logger.info(
                f"Compiled with 'adam' and set learning rate to {LEARNING_RATE}")
        except Exception as e2:
            logger.error(f"Failed to compile model: {e1}, {e2}")
            raise

    logger.info(f"✓ Model compiled (dropout={DROPOUT}, lr={LEARNING_RATE})")

    return model, tokenizer


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights to handle imbalance.

    Args:
        y: Array of labels

    Returns:
        Dictionary mapping class index to weight
    """
    # Map labels to 0, 1, 2
    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = np.array([label_map[label] for label in y])

    classes = np.unique(y_mapped)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_mapped
    )

    class_weights = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weights}")

    return class_weights


# =============================================================================
# TRAINING
# =============================================================================

def train_finbert(
    model: TFBertForSequenceClassification,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    class_weights: Optional[Dict] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE
) -> tf.keras.callbacks.History:
    """
    Train FinBERT model.

    Args:
        model: FinBERT model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        class_weights: Optional class weights
        epochs: Number of epochs
        batch_size: Batch size
        patience: Early stopping patience

    Returns:
        Training history
    """
    logger.info("=" * 80)
    logger.info("TRAINING FINBERT")
    logger.info("=" * 80)

    # Batch datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
    checkpoint_path = MODEL_DIR / 'finbert_best_model.h5'
    checkpoint = ModelCheckpoint(
        str(checkpoint_path),
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
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # Memory monitor
    if PSUTIL_AVAILABLE:
        callbacks.append(MemoryMonitor())

    # TensorBoard
    try:
        tensorboard = TensorBoard(
            log_dir=str(OUTPUT_DIR / 'logs'),
            histogram_freq=1
        )
        callbacks.append(tensorboard)
    except:
        pass

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

def evaluate_finbert(
    model: TFBertForSequenceClassification,
    test_dataset: tf.data.Dataset,
    y_test: np.ndarray,
    tokenizer: BertTokenizer
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
    """
    Evaluate FinBERT model.

    Returns:
        Tuple of (metrics_dict, y_pred, y_pred_proba, roc_curves)
    """
    logger.info("=" * 80)
    logger.info("EVALUATING FINBERT")
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

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred_mapped, labels=[-1, 0, 1], zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred_mapped, labels=[-1, 0, 1])

    # ROC AUC
    y_test_binarized = label_binarize(y_test, classes=[-1, 0, 1])
    # Reorder probabilities: [neg, neu, pos]
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
# VISUALIZATIONS
# =============================================================================

def plot_training_history(history, save_path: Optional[str] = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FinBERT Training History', fontsize=16, fontweight='bold')

    epochs = range(1, len(history.history['loss']) + 1)

    axes[0].plot(epochs, history.history['loss'], 'o-',
                 label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history.history['val_loss'], 's-',
                 label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.history['accuracy'], 'o-',
                 label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history.history['val_accuracy'], 's-',
                 label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path}")
    if COLAB_AVAILABLE:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None) -> None:
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
    plt.title('Confusion Matrix - FinBERT', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path}")
    if COLAB_AVAILABLE:
        plt.show()
    plt.close()


def plot_classification_report_table(metrics: Dict, save_path: Optional[str] = None) -> None:
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

    plt.title('Classification Report - FinBERT',
              fontsize=16, fontweight='bold', pad=20)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path}")
    if COLAB_AVAILABLE:
        plt.show()
    plt.close()


def plot_roc_curves(roc_curves: Dict, metrics: Dict, save_path: Optional[str] = None) -> None:
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
    plt.title('ROC Curves (One-vs-Rest) - FinBERT',
              fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path}")
    if COLAB_AVAILABLE:
        plt.show()
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run FinBERT fine-tuning pipeline."""
    logger.info("=" * 80)
    logger.info("FINBERT FINE-TUNING - PHASE 3")
    logger.info("=" * 80)

    # Configure GPU
    gpu_available = configure_t4_gpu()

    # Load data
    train_df, test_df, y_train, y_test = load_and_prepare_data(
        data_file='phase1_dataset.csv'
    )

    # Split training into train and validation
    train_df_split, val_df, y_train_split, y_val = train_test_split(
        train_df, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_SEED
    )

    # Load FinBERT
    model, tokenizer = load_finbert_model(num_labels=3)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_finbert_dataset(
        train_df_split['clean_text'], y_train_split, tokenizer
    )
    val_dataset = prepare_finbert_dataset(
        val_df['clean_text'], y_val, tokenizer
    )
    test_dataset = prepare_finbert_dataset(
        test_df['clean_text'], y_test, tokenizer
    )

    # Calculate class weights
    class_weights = calculate_class_weights(y_train_split)

    # Train model
    history = train_finbert(
        model, train_dataset, val_dataset,
        class_weights=class_weights
    )

    # Load best model
    best_model_path = MODEL_DIR / 'finbert_best_model.h5'
    if best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)

    # Evaluate
    metrics, y_pred, y_pred_proba, roc_curves = evaluate_finbert(
        model, test_dataset, y_test, tokenizer
    )

    # Save metrics
    metrics_path = OUTPUT_DIR / 'finbert_metrics.json'
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
    plot_classification_report_table(
        metrics,
        save_path=str(PLOTS_DIR / 'classification_report.png')
    )
    plot_roc_curves(
        roc_curves,
        metrics,
        save_path=str(PLOTS_DIR / 'roc_curves.png')
    )

    # Print summary
    logger.info("=" * 80)
    logger.info("FINBERT TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"✓ Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(
        f"✓ Negative F1: {metrics['per_class_metrics']['negative']['f1']:.4f}")
    logger.info(
        f"✓ Neutral F1: {metrics['per_class_metrics']['neutral']['f1']:.4f}")
    logger.info(
        f"✓ Positive F1: {metrics['per_class_metrics']['positive']['f1']:.4f}")
    logger.info(f"✓ Model saved to: {best_model_path}")
    logger.info(f"✓ Metrics saved to: {metrics_path}")
    logger.info(f"✓ Plots saved to: {PLOTS_DIR}")
    logger.info("=" * 80)

    return model, metrics, history


if __name__ == "__main__":
    main()
