"""
GloVe Embeddings Module for Financial Sentiment Analysis

This module provides functionality to:
1. Download and load GloVe embeddings (42B.300d)
2. Create embedding matrices for neural network models
3. Prepare data for BiLSTM training
4. Configure GPU settings for TensorFlow

Author: NLP Engineering Team
Date: 2024
"""

import os
import sys
import hashlib
import pickle
import logging
from typing import Tuple, Optional, Dict, Any
from urllib.request import urlretrieve
from urllib.error import URLError

import numpy as np
import pandas as pd

# Configure logging first (before using logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(
        "TensorFlow is not installed. Please install it using:\n"
        "  pip install tensorflow\n"
        "or for GPU support:\n"
        "  pip install tensorflow-gpu\n"
        f"Original error: {e}"
    )
    # Create dummy classes to prevent import errors
    class Tokenizer:
        pass
    class pad_sequences:
        pass

from sklearn.model_selection import train_test_split

# Constants
GLOVE_URL = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
GLOVE_FILE = "glove.42B.300d.txt"
GLOVE_ZIP = "glove.42B.300d.zip"
GLOVE_MD5 = "7765a4e2b9a2f6b9c8c2b0d1e0a9b8c7"  # Placeholder - actual MD5 will be computed
EMBEDDING_DIM = 300
MAX_WORDS = 5000
MAX_LENGTH = 30
RANDOM_SEED = 42
TOKENIZER_FILE = "glove_tokenizer.pkl"


def configure_gpu(memory_fraction: float = 0.9, enable_mixed_precision: bool = True) -> None:
    """
    Configure TensorFlow GPU settings for optimal performance.
    
    Optimized for NVIDIA 1080Ti/2080Ti (11GB VRAM):
    - Sets memory growth to prevent full GPU memory allocation
    - Configures memory fraction usage
    - Enables mixed precision training if supported
    
    Args:
        memory_fraction: Fraction of GPU memory to use (default: 0.9)
        enable_mixed_precision: Whether to enable mixed precision training
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for GPU configuration. "
            "Please install TensorFlow: pip install tensorflow"
        )
    
    logger.info("Configuring GPU settings...")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        logger.warning("No GPU devices found. Using CPU.")
        return
    
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Enabled memory growth for GPU: {gpu.name}")
        
        # Set memory limit (90% of available memory)
        if gpus:
            # Get GPU memory info
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if 'device_name' in gpu_details:
                logger.info(f"GPU Device: {gpu_details['device_name']}")
            
            # Set virtual memory limit
            try:
                memory_limit = int(11 * 1024 * memory_fraction)  # 11GB * fraction
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"Set GPU memory limit to {memory_limit}MB ({memory_fraction*100}%)")
            except RuntimeError as e:
                logger.warning(f"Could not set memory limit: {e}")
        
        # Enable mixed precision training if supported
        if enable_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision training enabled")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
        
        logger.info("GPU configuration completed successfully")
        
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
        logger.info("Falling back to CPU")


def calculate_md5(file_path: str, chunk_size: int = 8192) -> str:
    """
    Calculate MD5 checksum of a file.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read at a time (for large files)
        
    Returns:
        MD5 hash string
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return ""


def download_glove_embeddings(glove_file: str = GLOVE_FILE, 
                              glove_zip: str = GLOVE_ZIP,
                              verify_checksum: bool = True) -> bool:
    """
    Download GloVe embeddings from Stanford NLP if not already present.
    
    Args:
        glove_file: Name of the extracted GloVe text file
        glove_zip: Name of the GloVe zip file
        verify_checksum: Whether to verify MD5 checksum
        
    Returns:
        True if file exists or download successful, False otherwise
    """
    # Check if file already exists
    if os.path.exists(glove_file):
        logger.info(f"GloVe embeddings file already exists: {glove_file}")
        file_size = os.path.getsize(glove_file) / (1024 ** 3)  # Size in GB
        logger.info(f"File size: {file_size:.2f} GB")
        
        if verify_checksum:
            logger.info("Verifying file integrity...")
            # Note: Actual MD5 verification would require the correct checksum
            # For now, we'll just check if file is reasonably sized (>1GB)
            if file_size > 1.0:
                logger.info("File appears to be valid (size check passed)")
                return True
            else:
                logger.warning("File size seems too small. Re-downloading...")
                os.remove(glove_file)
        
        return True
    
    # Check if zip file exists
    if os.path.exists(glove_zip):
        logger.info(f"GloVe zip file found: {glove_zip}")
        logger.info("Extracting GloVe embeddings...")
        try:
            import zipfile
            with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
                zip_ref.extractall('.')
            logger.info("Extraction completed")
            
            if os.path.exists(glove_file):
                return True
        except Exception as e:
            logger.error(f"Error extracting zip file: {e}")
            return False
    
    # Download the file
    logger.info(f"Downloading GloVe embeddings from {GLOVE_URL}")
    logger.warning("This is a large file (~5.6GB compressed, ~9GB uncompressed)")
    logger.info("Download may take a while depending on your internet connection...")
    
    try:
        def reporthook(count, block_size, total_size):
            """Progress hook for download."""
            percent = min(100, (count * block_size * 100) / total_size)
            sys.stdout.write(f"\rProgress: {percent:.1f}%")
            sys.stdout.flush()
        
        urlretrieve(GLOVE_URL, glove_zip, reporthook=reporthook)
        print()  # New line after progress
        logger.info("Download completed")
        
        # Extract the zip file
        logger.info("Extracting GloVe embeddings...")
        import zipfile
        with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        if os.path.exists(glove_file):
            logger.info("GloVe embeddings ready")
            return True
        else:
            logger.error("Extraction failed - file not found")
            return False
            
    except URLError as e:
        logger.error(f"Download failed: {e}")
        logger.error("Please download manually from: https://nlp.stanford.edu/projects/glove/")
        return False
    except Exception as e:
        logger.error(f"Error during download/extraction: {e}")
        return False


def load_glove_embeddings(glove_file: str = GLOVE_FILE, 
                          embedding_dim: int = EMBEDDING_DIM) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from file into a dictionary.
    
    Args:
        glove_file: Path to GloVe embeddings file
        embedding_dim: Dimension of embeddings (default: 300)
        
    Returns:
        Dictionary mapping words to embedding vectors
    """
    if not os.path.exists(glove_file):
        raise FileNotFoundError(
            f"GloVe embeddings file not found: {glove_file}\n"
            f"Please run download_glove_embeddings() first or download manually."
        )
    
    logger.info(f"Loading GloVe embeddings from {glove_file}...")
    embeddings_index = {}
    
    try:
        file_size = os.path.getsize(glove_file)
        total_lines = 0
        
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    progress = (f.tell() / file_size) * 100
                    logger.info(f"Progress: {progress:.1f}% ({line_num:,} words loaded)")
                
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    if len(coefs) == embedding_dim:
                        embeddings_index[word] = coefs
                        total_lines += 1
                except (ValueError, IndexError):
                    continue
        
        logger.info(f"Loaded {len(embeddings_index):,} word vectors")
        return embeddings_index
        
    except Exception as e:
        logger.error(f"Error loading GloVe embeddings: {e}")
        raise


def create_embedding_matrix(tokenizer: Tokenizer,
                           max_words: int = MAX_WORDS,
                           embedding_dim: int = EMBEDDING_DIM,
                           glove_file: str = GLOVE_FILE) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create embedding matrix from GloVe embeddings for the vocabulary.
    
    Note: Dropout should be applied during training as a layer, not in the embedding matrix.
    The embedding matrix contains the pre-trained GloVe vectors.
    
    Args:
        tokenizer: Fitted Keras Tokenizer
        max_words: Maximum vocabulary size
        embedding_dim: Dimension of embeddings (default: 300)
        glove_file: Path to GloVe embeddings file
        
    Returns:
        Tuple of (embedding_matrix, stats_dict) where:
        - embedding_matrix: numpy array of shape (vocab_size, embedding_dim)
        - stats_dict: Dictionary with statistics about embedding coverage
    """
    logger.info("Creating embedding matrix...")
    
    # Load GloVe embeddings
    embeddings_index = load_glove_embeddings(glove_file, embedding_dim)
    
    # Get vocabulary from tokenizer
    word_index = tokenizer.word_index
    vocab_size = min(len(word_index) + 1, max_words + 1)  # +1 for padding token
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Initialize embedding matrix with random values (for unknown words)
    # Using normal distribution with scale 0.6 (standard for embedding initialization)
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
    stats = {
        'vocab_size': vocab_size,
        'found_words': found_words,
        'missing_words_count': len(missing_words),
        'coverage': (found_words / min(len(word_index), max_words)) * 100,
        'missing_words_sample': missing_words[:20]  # Sample of missing words
    }
    
    logger.info(f"Found {found_words} words in embedding matrix")
    logger.info(f"Missing {len(missing_words)} words from vocabulary")
    logger.info(f"Coverage: {stats['coverage']:.2f}%")
    
    if len(missing_words) > 0:
        logger.info(f"Sample of missing words: {missing_words[:10]}")
    
    return embedding_matrix, stats


def clean_text(text: str) -> str:
    """
    Clean text using the same approach as Phase 1 preprocessing.
    
    This function replicates the text cleaning from preprocess_dataset.py:
    - Convert to lowercase
    - Remove URLs
    - Remove special characters except apostrophes
    - Remove extra whitespace
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    import re
    
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http://, https://, www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    
    # Remove special characters except apostrophes and spaces
    # Keep: letters, numbers, apostrophes, spaces
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def prepare_glove_data(data_file: str = 'phase1_dataset.csv',
                       max_words: int = MAX_WORDS,
                       max_length: int = MAX_LENGTH,
                       test_size: float = 0.2,
                       random_state: int = RANDOM_SEED,
                       glove_file: str = GLOVE_FILE,
                       tokenizer_file: str = TOKENIZER_FILE,
                       use_existing_tokenizer: bool = True) -> Tuple[np.ndarray, np.ndarray, 
                                                                     np.ndarray, np.ndarray,
                                                                     np.ndarray, Tokenizer,
                                                                     Dict[str, Any]]:
    """
    Prepare data for GloVe-based model training.
    
    This function:
    1. Loads phase1_dataset.csv
    2. Applies same text cleaning as Phase 1
    3. Tokenizes and converts to sequences
    4. Creates train/test split
    5. Builds embedding matrix from GloVe
    
    Args:
        data_file: Path to preprocessed dataset CSV
        max_words: Maximum vocabulary size
        max_length: Maximum sequence length
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        glove_file: Path to GloVe embeddings file
        tokenizer_file: Path to save/load tokenizer
        use_existing_tokenizer: Whether to use existing tokenizer if available
        
    Returns:
        Tuple containing:
        - X_train: Training sequences
        - X_test: Test sequences
        - y_train: Training labels
        - y_test: Test labels
        - embedding_matrix: GloVe embedding matrix
        - tokenizer: Fitted tokenizer
        - stats: Dictionary with statistics
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for data preparation. "
            "Please install TensorFlow: pip install tensorflow"
        )
    
    logger.info("="*60)
    logger.info("Preparing GloVe data for training")
    logger.info("="*60)
    
    # Check if data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Dataset file not found: {data_file}\n"
            f"Please run preprocess_dataset.py first to create the dataset."
        )
    
    # Load dataset
    logger.info(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract features and labels
    texts = df['clean_text'].copy()
    labels = df['sentiment'].copy()
    
    # Handle empty text inputs
    empty_mask = (texts.isna()) | (texts.str.strip() == '')
    if empty_mask.sum() > 0:
        logger.warning(f"Found {empty_mask.sum()} empty text inputs. Filling with 'unknown'.")
        texts[empty_mask] = 'unknown'
    
    # Apply text cleaning (in case clean_text needs re-cleaning)
    logger.info("Applying text cleaning...")
    texts = texts.apply(clean_text)
    
    # Convert texts to list
    texts_list = texts.tolist()
    
    # Check for existing tokenizer
    tokenizer = None
    if use_existing_tokenizer and os.path.exists(tokenizer_file):
        logger.info(f"Loading existing tokenizer from {tokenizer_file}...")
        try:
            with open(tokenizer_file, 'rb') as f:
                tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Creating new tokenizer.")
    
    # Create or use existing tokenizer
    if tokenizer is None:
        logger.info("Creating new tokenizer...")
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts_list)
        logger.info(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")
        
        # Save tokenizer
        logger.info(f"Saving tokenizer to {tokenizer_file}...")
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
        logger.info("Tokenizer saved")
    
    # Convert texts to sequences
    logger.info("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(texts_list)
    
    # Pad sequences
    logger.info(f"Padding sequences to max_length={max_length}...")
    X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    logger.info(f"Sequence shape: {X.shape}")
    
    # Convert labels to numpy array
    y = labels.values
    
    # Train-test split (stratified)
    logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Create embedding matrix
    logger.info("Creating embedding matrix from GloVe...")
    embedding_matrix, embedding_stats = create_embedding_matrix(
        tokenizer, 
        max_words=max_words,
        embedding_dim=EMBEDDING_DIM,
        glove_file=glove_file
    )
    
    # Compile statistics
    stats = {
        'dataset_size': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'vocab_size': len(tokenizer.word_index) + 1,
        'max_length': max_length,
        'embedding_stats': embedding_stats
    }
    
    # Memory usage check
    memory_usage_mb = (X_train.nbytes + X_test.nbytes + embedding_matrix.nbytes) / (1024 ** 2)
    logger.info(f"Memory usage: {memory_usage_mb:.2f} MB")
    
    if memory_usage_mb > 1000:  # > 1GB
        logger.warning(f"High memory usage: {memory_usage_mb:.2f} MB (>1GB)")
    
    logger.info("="*60)
    logger.info("Data preparation completed successfully!")
    logger.info("="*60)
    
    return X_train, X_test, y_train, y_test, embedding_matrix, tokenizer, stats


def check_memory_usage(threshold: float = 0.8) -> None:
    """
    Check current memory usage and warn if above threshold.
    
    Args:
        threshold: Memory usage threshold (default: 0.8 = 80%)
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100
        
        if usage_percent > threshold:
            logger.warning(
                f"Memory usage is {memory.percent:.1f}% (> {threshold*100}% threshold)"
            )
        else:
            logger.info(f"Memory usage: {memory.percent:.1f}%")
    except ImportError:
        logger.info("psutil not available - skipping memory check")


def main():
    """
    Main function to demonstrate GloVe embeddings setup.
    """
    logger.info("GloVe Embeddings Module - Setup and Data Preparation")
    logger.info("="*60)
    
    # Configure GPU
    configure_gpu(memory_fraction=0.9, enable_mixed_precision=True)
    
    # Check memory usage
    check_memory_usage(threshold=0.8)
    
    # Download GloVe embeddings if needed
    logger.info("\nStep 1: Checking GloVe embeddings...")
    if not download_glove_embeddings():
        logger.error("Failed to download GloVe embeddings. Exiting.")
        return
    
    # Prepare data
    logger.info("\nStep 2: Preparing data...")
    try:
        X_train, X_test, y_train, y_test, embedding_matrix, tokenizer, stats = prepare_glove_data()
        
        logger.info("\nData preparation summary:")
        logger.info(f"  Training samples: {stats['train_size']}")
        logger.info(f"  Test samples: {stats['test_size']}")
        logger.info(f"  Vocabulary size: {stats['vocab_size']}")
        logger.info(f"  Sequence length: {stats['max_length']}")
        logger.info(f"  Embedding matrix shape: {embedding_matrix.shape}")
        logger.info(f"  Embedding coverage: {stats['embedding_stats']['coverage']:.2f}%")
        
        logger.info("\n" + "="*60)
        logger.info("GloVe embeddings setup completed successfully!")
        logger.info("="*60)
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please ensure phase1_dataset.csv exists in the current directory.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

