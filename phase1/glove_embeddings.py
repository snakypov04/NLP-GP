"""
=============================================================================
GloVe Embeddings Module for Financial Sentiment Analysis
=============================================================================
Standalone Python module (no Google Drive dependency)

This module provides functionality to:
1. Download and load GloVe embeddings (42B.300d with 6B fallback)
2. Create embedding matrices for neural network models
3. Prepare data for BiLSTM training
4. Configure GPU settings for TensorFlow

Optimized for: Google Colab T4 GPU (16GB VRAM)
Author: NLP Engineering Team
Date: 2024
=============================================================================
"""

from sklearn.model_selection import train_test_split
import os
import sys
import gc
import pickle
import hashlib
import zipfile
import requests
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
    logger.info(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger.error(f"TensorFlow not available: {e}")
    raise ImportError(
        "TensorFlow is required. Install with: pip install tensorflow")


# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

# =============================================================================
# CONSTANTS
# =============================================================================

GLOVE_CONFIGS = {
    '42B': {
        'url': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'filename': 'glove.42B.300d.txt',
        'zip_name': 'glove.42B.300d.zip',
        'dim': 300,
        'vocab_size': 1900000,
        'size_gb': 5.6,
    },
    '6B': {  # Fallback option
        'url': 'http://nlp.stanford.edu/data/glove.6B.zip',
        'filename': 'glove.6B.300d.txt',
        'zip_name': 'glove.6B.zip',
        'dim': 300,
        'vocab_size': 400000,
        'size_gb': 0.8,
    }
}

EMBEDDING_DIM = 300
MAX_WORDS = 5000
MAX_LENGTH = 30
RANDOM_SEED = 42

# Local paths (current directory)
BASE_PATH = Path('.')
EMBEDDINGS_PATH = BASE_PATH / 'embeddings'
DATA_PATH = BASE_PATH / 'data'
TOKENIZER_FILE = 'glove_tokenizer.pkl'
EMBEDDING_MATRIX_FILE = 'glove_embedding_matrix.npy'
METADATA_FILE = 'glove_metadata.pkl'

# Create directories
EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MEMORY MONITORING UTILITIES
# =============================================================================

def get_memory_usage() -> Dict:
    """Get current memory usage statistics."""
    if not PSUTIL_AVAILABLE:
        return {'available': True}

    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'used_gb': mem.used / (1024**3),
        'available_gb': mem.available / (1024**3),
        'percent': mem.percent
    }


def print_memory_status(label: str = "") -> None:
    """Print formatted memory status."""
    if not PSUTIL_AVAILABLE:
        return

    mem = get_memory_usage()
    logger.info("=" * 80)
    logger.info(f"MEMORY STATUS {f'- {label}' if label else ''}")
    logger.info("=" * 80)
    logger.info(f"Total:     {mem['total_gb']:.2f} GB")
    logger.info(f"Used:      {mem['used_gb']:.2f} GB ({mem['percent']:.1f}%)")
    logger.info(f"Available: {mem['available_gb']:.2f} GB")

    if mem['percent'] > 90:
        logger.warning("🚨 CRITICAL: Memory usage above 90% - risk of crash!")
    elif mem['percent'] > 80:
        logger.warning("⚠️  WARNING: Memory usage above 80%!")
    else:
        logger.info("✓ Memory usage is healthy")


def clear_memory() -> None:
    """Clear memory and run garbage collection."""
    gc.collect()
    if TENSORFLOW_AVAILABLE and tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    logger.info("✓ Memory cleared")


# =============================================================================
# GPU CONFIGURATION
# =============================================================================

def configure_gpu(memory_fraction: float = 0.9, enable_mixed_precision: bool = True) -> bool:
    """
    Configure TensorFlow GPU settings for optimal performance.

    Optimized for T4 GPU (16GB VRAM):
    - Sets memory growth to prevent full GPU memory allocation
    - Enables mixed precision training (FP16)
    - Enables XLA JIT compilation

    Args:
        memory_fraction: Fraction of GPU memory to use (default: 0.9)
        enable_mixed_precision: Whether to enable mixed precision training

    Returns:
        True if GPU is available and configured, False otherwise
    """
    logger.info("=" * 80)
    logger.info("GPU CONFIGURATION")
    logger.info("=" * 80)

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

            logger.info(f"✓ Memory growth enabled for {gpu.name}")

        # Enable mixed precision training
        if enable_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info(
                    "✓ Mixed precision (FP16) enabled for faster training")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")

        # Enable XLA optimization
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("✓ XLA JIT compilation enabled")
        except Exception as e:
            logger.warning(f"Could not enable XLA: {e}")

        return True

    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
        logger.info("Falling back to CPU")
        return False


# =============================================================================
# GLOVE DOWNLOAD
# =============================================================================

def download_file_with_progress(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download file with progress bar and retry logic.

    Args:
        url: Download URL
        destination: Local file path
        chunk_size: Download chunk size in bytes

    Returns:
        True if successful, False otherwise
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")

            # Send request with streaming
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get file size
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)

            logger.info(f"✓ Download completed: {destination.name}")
            return True

        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                logger.error("All retry attempts exhausted")
                return False

    return False


def verify_file_integrity(filepath: Path, expected_size_gb: float = None) -> bool:
    """
    Verify downloaded file integrity.

    Args:
        filepath: Path to file
        expected_size_gb: Expected file size in GB (approximate)

    Returns:
        True if file appears valid
    """
    if not filepath.exists():
        return False

    actual_size_gb = filepath.stat().st_size / (1024**3)
    logger.info(f"File size: {actual_size_gb:.2f} GB")

    if expected_size_gb:
        # Allow 10% tolerance
        if actual_size_gb < expected_size_gb * 0.9:
            logger.warning(
                f"File seems too small (expected ~{expected_size_gb:.2f} GB)")
            return False

    logger.info("✓ File integrity check passed")
    return True


def download_and_extract_glove(config_name: str = '42B') -> Optional[Path]:
    """
    Download and extract GloVe embeddings.

    Args:
        config_name: '42B' or '6B' (fallback)

    Returns:
        Path to extracted GloVe file, or None if failed
    """
    logger.info("=" * 80)
    logger.info(f"GLOVE {config_name} EMBEDDINGS DOWNLOAD")
    logger.info("=" * 80)

    config = GLOVE_CONFIGS[config_name]
    glove_file = EMBEDDINGS_PATH / config['filename']
    zip_file = EMBEDDINGS_PATH / config['zip_name']

    # Check if already exists
    if glove_file.exists():
        logger.info(f"✓ GloVe file already exists: {glove_file.name}")
        if verify_file_integrity(glove_file, config['size_gb']):
            return glove_file
        else:
            logger.warning("File appears corrupted, re-downloading...")
            glove_file.unlink()

    # Check available memory before download
    if PSUTIL_AVAILABLE:
        mem = get_memory_usage()
        if mem['available_gb'] < 2.0:
            logger.warning(
                f"Low memory available ({mem['available_gb']:.2f} GB)")
            logger.info("Clearing memory before download...")
            clear_memory()

    # Download zip file
    if not zip_file.exists():
        logger.info(
            f"📥 Downloading GloVe {config_name} ({config['size_gb']:.1f} GB compressed)")
        logger.info(f"URL: {config['url']}")
        logger.info(
            "This may take 5-15 minutes depending on connection speed...")

        if not download_file_with_progress(config['url'], zip_file):
            logger.error("Download failed")
            return None
    else:
        logger.info(f"✓ Zip file already downloaded: {zip_file.name}")

    # Extract zip file
    logger.info(f"📦 Extracting {zip_file.name}...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            target_file = config['filename']

            if target_file not in file_list:
                logger.info(f"Available files: {file_list}")
                # Try to find correct file
                matching = [f for f in file_list if f.endswith(
                    '.txt') and '300d' in f]
                if matching:
                    target_file = matching[0]
                    logger.info(f"Using: {target_file}")

            # Extract
            logger.info(f"Extracting: {target_file}")
            zip_ref.extract(target_file, EMBEDDINGS_PATH)

            # Rename if necessary
            extracted_path = EMBEDDINGS_PATH / target_file
            if extracted_path != glove_file:
                extracted_path.rename(glove_file)

        logger.info(f"✓ Extraction completed: {glove_file.name}")

        # Verify extracted file
        if verify_file_integrity(glove_file):
            # Clean up zip file to save space
            logger.info("Cleaning up zip file...")
            zip_file.unlink()
            logger.info("✓ Cleanup completed")
            return glove_file
        else:
            return None

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None


# =============================================================================
# TEXT PREPROCESSING (PHASE 1 CONSISTENCY)
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean text using EXACT Phase 1 preprocessing approach.
    Must match preprocess_dataset.py to ensure consistency.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text == '':
        return ''

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http://, https://, www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    # Remove special characters except apostrophes and spaces
    text = re.sub(r"[^a-z0-9'\s]", '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# =============================================================================
# MEMORY-EFFICIENT EMBEDDING LOADER
# =============================================================================

def load_glove_for_vocabulary(
    glove_file: Path,
    vocabulary: set,
    embedding_dim: int = 300
) -> Dict[str, np.ndarray]:
    """
    Load only the GloVe embeddings needed for our vocabulary.
    This prevents loading all 1.9M embeddings into memory.

    Args:
        glove_file: Path to GloVe embeddings file
        vocabulary: Set of words in our vocabulary
        embedding_dim: Embedding dimension (default: 300)

    Returns:
        Dictionary mapping words to embedding vectors
    """
    logger.info("=" * 80)
    logger.info("LOADING GLOVE EMBEDDINGS (VOCABULARY-FILTERED)")
    logger.info("=" * 80)
    logger.info(f"Vocabulary size: {len(vocabulary):,} words")
    logger.info(f"Embedding dimension: {embedding_dim}")

    embeddings_index = {}
    found_count = 0
    total_lines = 0

    # Get file size for progress tracking
    file_size = glove_file.stat().st_size
    logger.info(f"Reading GloVe file: {glove_file.name}")
    logger.info(f"File size: {file_size / (1024**3):.2f} GB")
    logger.info("Loading only words in vocabulary...")

    try:
        with open(glove_file, 'r', encoding='utf-8') as f:
            # Create progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading") as pbar:
                for line in f:
                    total_lines += 1
                    pbar.update(len(line.encode('utf-8')))

                    # Parse line
                    values = line.rstrip().split(' ')
                    word = values[0]

                    # Only load if word is in our vocabulary
                    if word in vocabulary:
                        try:
                            coefs = np.asarray(values[1:], dtype='float32')
                            if len(coefs) == embedding_dim:
                                embeddings_index[word] = coefs
                                found_count += 1
                        except (ValueError, IndexError):
                            continue

                    # Periodic progress update
                    if total_lines % 100000 == 0:
                        pbar.set_postfix({
                            'found': f"{found_count}/{len(vocabulary)}",
                            'coverage': f"{(found_count/len(vocabulary)*100):.1f}%"
                        })

        coverage = (found_count / len(vocabulary)) * 100

        logger.info(f"✓ Loading completed:")
        logger.info(f"   Total GloVe vectors processed: {total_lines:,}")
        logger.info(
            f"   Vocabulary words found: {found_count:,} / {len(vocabulary):,}")
        logger.info(f"   Coverage: {coverage:.2f}%")
        logger.info(f"   Missing words: {len(vocabulary) - found_count:,}")

        if coverage < 70:
            logger.warning("Low coverage - consider checking preprocessing")

        return embeddings_index

    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise


# =============================================================================
# EMBEDDING MATRIX CREATION
# =============================================================================

def create_embedding_matrix(
    tokenizer: Tokenizer,
    embeddings_index: Dict[str, np.ndarray],
    max_words: int = MAX_WORDS,
    embedding_dim: int = EMBEDDING_DIM
) -> Tuple[np.ndarray, Dict]:
    """
    Create embedding matrix from GloVe embeddings for the vocabulary.

    Args:
        tokenizer: Fitted Keras Tokenizer
        embeddings_index: Dictionary of word -> embedding vector
        max_words: Maximum vocabulary size
        embedding_dim: Embedding dimension

    Returns:
        Tuple of (embedding_matrix, stats_dict)
    """
    logger.info("Creating embedding matrix...")

    # Get vocabulary from tokenizer
    word_index = tokenizer.word_index
    vocab_size = min(len(word_index) + 1, max_words +
                     1)  # +1 for padding token

    logger.info(f"Vocabulary size: {vocab_size:,}")
    logger.info(f"Embedding dimension: {embedding_dim}")

    # Initialize embedding matrix with random values (for unknown words)
    embedding_matrix = np.random.normal(
        scale=0.6,
        size=(vocab_size, embedding_dim)
    ).astype('float32')

    # Track statistics
    found_words = 0
    missing_words = []

    # Fill embedding matrix with GloVe vectors
    for word, i in word_index.items():
        if i >= vocab_size:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words += 1
        else:
            missing_words.append(word)

    # Statistics
    coverage = (found_words / min(len(word_index), max_words)) * 100
    stats = {
        'vocab_size': vocab_size,
        'found_words': found_words,
        'missing_words_count': len(missing_words),
        'coverage': coverage,
        'missing_words_sample': missing_words[:20]
    }

    logger.info(f"✓ Embedding matrix created:")
    logger.info(f"   Shape: {embedding_matrix.shape}")
    logger.info(
        f"   Found words: {found_words:,} / {min(len(word_index), max_words):,}")
    logger.info(f"   Coverage: {coverage:.2f}%")
    logger.info(
        f"   Memory usage: {embedding_matrix.nbytes / (1024**2):.2f} MB")

    if len(missing_words) > 0:
        logger.info(f"   Sample of missing words: {missing_words[:10]}")

    return embedding_matrix, stats


# =============================================================================
# MAIN DATA PREPARATION FUNCTION
# =============================================================================

def prepare_glove_data(
    data_file: str = 'phase1_dataset.csv',
    max_words: int = MAX_WORDS,
    max_length: int = MAX_LENGTH,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    glove_config: str = '42B',
    use_cached: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer, Dict]:
    """
    Prepare complete dataset with GloVe embeddings.

    This function:
    1. Loads phase1_dataset.csv
    2. Applies same text cleaning as Phase 1
    3. Tokenizes and converts to sequences
    4. Creates train/test split
    5. Builds embedding matrix from GloVe

    Args:
        data_file: Path to preprocessed dataset CSV
        max_words: Maximum vocabulary size (default: 5000)
        max_length: Maximum sequence length (default: 30)
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        glove_config: '42B' or '6B' (default: '42B')
        use_cached: Whether to use cached tokenizer/embeddings (default: True)

    Returns:
        Tuple containing:
        - X_train: Training sequences
        - X_test: Test sequences
        - y_train: Training labels
        - y_test: Test labels
        - embedding_matrix: GloVe embedding matrix
        - tokenizer: Fitted tokenizer
        - metadata: Dictionary with statistics
    """
    logger.info("=" * 80)
    logger.info("DATA PREPARATION WITH GLOVE EMBEDDINGS")
    logger.info("=" * 80)

    # Paths for cached objects
    tokenizer_path = DATA_PATH / TOKENIZER_FILE
    embedding_matrix_path = DATA_PATH / EMBEDDING_MATRIX_FILE
    metadata_path = DATA_PATH / METADATA_FILE

    # Check for cached objects
    if use_cached and all(p.exists() for p in [tokenizer_path, embedding_matrix_path, metadata_path]):
        logger.info("✓ Found cached tokenizer and embeddings")
        logger.info("Loading from cache...")

        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            embedding_matrix = np.load(embedding_matrix_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            logger.info("✓ Loaded from cache:")
            logger.info(
                f"   Tokenizer vocabulary: {len(tokenizer.word_index):,}")
            logger.info(f"   Embedding matrix shape: {embedding_matrix.shape}")
            logger.info(f"   Coverage: {metadata['coverage']:.2f}%")

            # Still need to load and split data
            logger.info(f"\nLoading dataset: {data_file}")
            df = pd.read_csv(data_file)
            texts = df['clean_text'].fillna('').apply(clean_text).tolist()
            labels = df['sentiment'].values

            # Create sequences
            sequences = tokenizer.texts_to_sequences(texts)
            X = pad_sequences(sequences, maxlen=max_length,
                              padding='post', truncating='post')
            y = labels

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            logger.info(f"✓ Training samples: {len(X_train):,}")
            logger.info(f"✓ Test samples: {len(X_test):,}")

            return X_train, X_test, y_train, y_test, embedding_matrix, tokenizer, metadata

        except Exception as e:
            logger.warning(f"Cache loading failed: {e}")
            logger.info("Regenerating from scratch...")

    # Load dataset
    logger.info(f"\n📊 Loading dataset: {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            f"Please ensure phase1_dataset.csv exists in the current directory."
        )

    df = pd.read_csv(data_file)
    logger.info(f"✓ Loaded {len(df):,} samples")

    # Print initial memory status
    print_memory_status("After loading dataset")

    # Extract and clean texts
    logger.info("\n🧹 Cleaning texts...")
    texts = df['clean_text'].fillna('').apply(clean_text).tolist()

    # Handle empty texts
    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count > 0:
        logger.warning(
            f"Found {empty_count} empty texts - filling with 'unknown'")
        texts = ['unknown' if not t.strip() else t for t in texts]

    # Create tokenizer
    logger.info(f"\n🔤 Creating tokenizer (max_words={max_words:,})...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)

    vocab_size = len(tokenizer.word_index)
    actual_vocab_size = min(vocab_size + 1, max_words + 1)  # +1 for padding

    logger.info(f"✓ Tokenizer created:")
    logger.info(f"   Total unique words: {vocab_size:,}")
    logger.info(f"   Vocabulary size (with limit): {actual_vocab_size:,}")

    # Get vocabulary set for filtering
    vocabulary = set(
        word for word, idx in tokenizer.word_index.items() if idx < max_words)
    logger.info(f"   Vocabulary to match in GloVe: {len(vocabulary):,} words")

    # Download and load GloVe
    glove_file = download_and_extract_glove(glove_config)
    if glove_file is None:
        if glove_config == '42B':
            logger.warning("GloVe 42B download failed, trying 6B fallback...")
            glove_file = download_and_extract_glove('6B')
            glove_config = '6B'

        if glove_file is None:
            raise RuntimeError("Failed to download GloVe embeddings")

    # Load embeddings for vocabulary only
    print_memory_status("Before loading embeddings")
    embeddings_index = load_glove_for_vocabulary(
        glove_file, vocabulary, EMBEDDING_DIM)
    print_memory_status("After loading embeddings")

    # Create embedding matrix
    embedding_matrix, embedding_stats = create_embedding_matrix(
        tokenizer, embeddings_index, max_words, EMBEDDING_DIM
    )

    # Prepare metadata
    metadata = {
        'vocab_size': actual_vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'max_length': max_length,
        'max_words': max_words,
        'coverage': embedding_stats['coverage'],
        'found_words': embedding_stats['found_words'],
        'missing_words_count': embedding_stats['missing_words_count'],
        'glove_config': glove_config,
        'dataset_size': len(df)
    }

    # Save to cache
    logger.info(f"\n💾 Saving to cache...")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    np.save(embedding_matrix_path, embedding_matrix)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"✓ Cached for future use")

    # Clean up embeddings_index to free memory
    del embeddings_index
    clear_memory()

    # Create sequences
    logger.info(f"\n📝 Creating sequences...")
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_length,
                      padding='post', truncating='post')
    y = df['sentiment'].values

    logger.info(f"✓ Sequences created: {X.shape}")

    # Train-test split (stratified)
    logger.info(
        f"\n✂️  Splitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(f"✓ Training samples: {len(X_train):,}")
    logger.info(f"✓ Test samples: {len(X_test):,}")
    logger.info(f"   Positive samples (train): {(y_train == 1).sum():,}")
    logger.info(f"   Negative samples (train): {(y_train == 0).sum():,}")

    print_memory_status("Final")

    return X_train, X_test, y_train, y_test, embedding_matrix, tokenizer, metadata


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to demonstrate GloVe embeddings setup.
    """
    logger.info("=" * 80)
    logger.info("GLOVE EMBEDDINGS MODULE - SETUP AND DATA PREPARATION")
    logger.info("=" * 80)

    # Configure GPU
    gpu_available = configure_gpu(
        memory_fraction=0.9, enable_mixed_precision=True)

    # Initial memory check
    print_memory_status("Initial")

    # Prepare data with GloVe embeddings
    try:
        X_train, X_test, y_train, y_test, embedding_matrix, tokenizer, metadata = prepare_glove_data(
            data_file='phase1_dataset.csv',
            max_words=MAX_WORDS,
            max_length=MAX_LENGTH,
            test_size=0.2,
            random_state=RANDOM_SEED,
            glove_config='42B',
            use_cached=True
        )

        # Print summary
        logger.info("=" * 80)
        logger.info("SETUP COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Training samples: {len(X_train):,}")
        logger.info(f"✓ Test samples: {len(X_test):,}")
        logger.info(f"✓ Embedding matrix shape: {embedding_matrix.shape}")
        logger.info(f"✓ Vocabulary size: {metadata['vocab_size']:,}")
        logger.info(f"✓ GloVe coverage: {metadata['coverage']:.2f}%")
        logger.info(f"✓ GPU available: {gpu_available}")
        logger.info("=" * 80)
        logger.info("✓ Data is ready for model training!")

        return X_train, X_test, y_train, y_test, embedding_matrix, tokenizer, metadata

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise
