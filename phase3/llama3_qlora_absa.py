"""
=============================================================================
QLoRA Fine-tuning for Llama 3 8B - Aspect-Based Sentiment Analysis (ABSA)
=============================================================================

Phase 3: QLoRA fine-tuning of Llama 3 8B for ABSA task
Optimized for GPU with limited VRAM (16GB+ recommended)

Features:
- Llama 3 8B with 4-bit quantization (QLoRA)
- LoRA adapters for efficient fine-tuning
- Fine-tuned on ABSA dataset with 'Target' and 'Other' tokens
- Memory-efficient training
- Comprehensive evaluation and comparison with DistilRoBERTa (87.4% accuracy)

Expected Performance: 88-92% accuracy

IMPORTANT: 
1. Install dependencies: pip install -r requirements.txt
2. For Llama 3, you may need Hugging Face token: export HF_TOKEN=your_token
3. Requires CUDA-capable GPU with 16GB+ VRAM

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
from typing import Dict, Tuple, Optional, List
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

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import logging as hf_logging

warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
MAX_LENGTH = 256  # Longer context for Llama 3
BATCH_SIZE = 4    # Small batch for QLoRA (adjust based on VRAM)
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 4 * 8 = 32
LEARNING_RATE = 2e-4  # Higher LR for LoRA
EPOCHS = 3
PATIENCE = 2
WARMUP_STEPS = 100
LOGGING_STEPS = 50
SAVE_STEPS = 500

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj", "k_proj",
                  "o_proj"]  # Llama 3 attention modules

# Paths
OUTPUT_DIR = Path('llama3_qlora_absa_results')
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
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# =============================================================================
# GPU CONFIGURATION
# =============================================================================


def configure_gpu():
    """Configure GPU and check availability."""
    logger.info("=" * 80)
    logger.info("GPU CONFIGURATION")
    logger.info("=" * 80)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow on CPU.")
        return False, None

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    logger.info(f"✓ GPU detected: {gpu_name}")
    logger.info(f"✓ GPU Memory: {gpu_memory:.2f} GB")

    # Clear cache
    torch.cuda.empty_cache()

    return True, device

# =============================================================================
# DATA LOADING AND PREPROCESSING
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

    def replace_entities_in_text(text: str, target_entity: str, all_entities_in_row: set) -> str:
        """Replace entities with 'Target' and 'Other' placeholders."""
        if pd.isna(text) or text == '':
            return ''
        text = str(text)

        other_entities = all_entities_in_row - {target_entity}

        # Replace other entities first
        for entity in other_entities:
            if entity and entity in text:
                pattern = r'\b' + re.escape(entity) + r'\b'
                text = re.sub(pattern, 'Other', text, flags=re.IGNORECASE)

        # Replace target entity
        if target_entity and target_entity in text:
            pattern = r'\b' + re.escape(target_entity) + r'\b'
            text = re.sub(pattern, 'Target', text, flags=re.IGNORECASE)

        return text

    # Get all entities per headline
    headline_to_entities = {}
    for _, row in df.iterrows():
        headline = row['headline']
        target = row['target_entity']
        if headline not in headline_to_entities:
            headline_to_entities[headline] = set()
        headline_to_entities[headline].add(target)

    # Create transformer-friendly text
    def create_transformer_text(row):
        headline = row['headline']
        target_entity = row['target_entity']
        entities_in_headline = headline_to_entities.get(
            headline, {target_entity})

        text = replace_entities_in_text(
            headline, target_entity, entities_in_headline)

        # Minimal cleaning
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text if text else 'unknown'

    df['text_for_transformer'] = df.apply(create_transformer_text, axis=1)

    # Handle empty texts
    empty_mask = df['text_for_transformer'].str.strip() == ''
    if empty_mask.sum() > 0:
        logger.warning(
            f"Found {empty_mask.sum()} empty texts - filling with 'unknown'")
        df.loc[empty_mask, 'text_for_transformer'] = 'unknown'

    logger.info("✓ Created transformer-friendly text")

    # Get labels
    y = df['sentiment'].values

    # Split data
    df_temp, test_df, y_temp, y_test = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=random_state
    )

    val_size_adjusted = val_size / (train_size + val_size)
    train_df, val_df, y_train, y_val = train_test_split(
        df_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )

    logger.info(f"✓ Training samples: {len(train_df):,}")
    logger.info(f"✓ Validation samples: {len(val_df):,}")
    logger.info(f"✓ Test samples: {len(test_df):,}")

    return train_df, val_df, test_df, y_train, y_val, y_test

# =============================================================================
# DATASET CLASS
# =============================================================================


class ABSADataset(Dataset):
    """Dataset class for ABSA task with Llama 3."""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Map labels: -1 -> 0 (negative), 0 -> 1 (neutral), 1 -> 2 (positive)
        self.label_map = {-1: 0, 0: 1, 1: 2}
        self.label_names = {0: "negative", 1: "neutral", 2: "positive"}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        mapped_label = self.label_map[label]
        label_name = self.label_names[mapped_label]

        # Create prompt for Llama 3
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a sentiment analysis expert. Analyze the sentiment towards the Target entity in the given text. Respond with only one word: negative, neutral, or positive.<|eot_id|><|start_header_id|>user<|end_header_id|>

Text: {text}

Sentiment towards Target: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label_name}<|eot_id|>"""

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # For causal LM, labels = input_ids
            'labels': encoding['input_ids'].flatten()
        }

# =============================================================================
# MODEL SETUP
# =============================================================================


def load_llama3_model_and_tokenizer():
    """
    Load Llama 3 8B with 4-bit quantization and prepare for QLoRA.

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("=" * 80)
    logger.info("LOADING LLAMA 3 8B WITH QLORA")
    logger.info("=" * 80)

    model_name = "meta-llama/Meta-Llama-3-8B"

    # Check for Hugging Face token
    hf_token = os.environ.get(
        'HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        logger.warning(
            "HF_TOKEN not found. You may need to set it for Llama 3 access.")
        logger.info("Set it with: export HF_TOKEN=your_token")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading {model_name} with 4-bit quantization...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("✓ Tokenizer loaded")

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    logger.info("✓ Model loaded with 4-bit quantization")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    logger.info("✓ Model prepared for k-bit training")

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    logger.info("✓ LoRA adapters applied")
    logger.info(
        f"  LoRA R: {LORA_R}, Alpha: {LORA_ALPHA}, Dropout: {LORA_DROPOUT}")

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer

# =============================================================================
# TRAINING
# =============================================================================


def train_llama3_qlora(
    model,
    tokenizer,
    train_dataset: ABSADataset,
    val_dataset: ABSADataset,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
):
    """Train Llama 3 with QLoRA."""
    logger.info("=" * 80)
    logger.info("TRAINING LLAMA 3 WITH QLORA")
    logger.info("=" * 80)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=WARMUP_STEPS,
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Custom data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(str(MODEL_DIR))
    logger.info(f"✓ Model saved to {MODEL_DIR}")

    # Save training metrics
    train_metrics = train_result.metrics
    logger.info(f"Training loss: {train_metrics['train_loss']:.4f}")

    return trainer, train_metrics

# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_llama3_qlora(
    model,
    tokenizer,
    test_dataset: ABSADataset,
    y_test: np.ndarray
) -> Dict:
    """Evaluate Llama 3 QLoRA model."""
    logger.info("=" * 80)
    logger.info("EVALUATING LLAMA 3 QLORA MODEL")
    logger.info("=" * 80)

    model.eval()
    device = next(model.parameters()).device

    all_predictions = []
    all_labels = []

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    label_map = {-1: 0, 0: 1, 1: 2}
    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    name_to_label = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode predictions
            for i, output in enumerate(outputs):
                # Extract generated text
                generated_text = tokenizer.decode(
                    output, skip_special_tokens=False)

                # Extract sentiment from assistant response
                if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                    assistant_response = generated_text.split(
                        "<|start_header_id|>assistant<|end_header_id|>")[-1]
                    assistant_response = assistant_response.split("<|eot_id|>")[
                        0].strip().lower()

                    # Map to label
                    if "negative" in assistant_response:
                        pred_label = 0
                    elif "positive" in assistant_response:
                        pred_label = 2
                    else:
                        pred_label = 1  # neutral
                else:
                    pred_label = 1  # default to neutral

                all_predictions.append(pred_label)

            # Get true labels
            true_labels = [label_map[int(y_test[batch_idx * BATCH_SIZE + i])]
                           for i in range(len(batch['input_ids']))]
            all_labels.extend(true_labels)

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Convert predictions back to original label space (-1, 0, 1)
    y_pred = np.array([name_to_label[p] for p in all_predictions])
    y_true = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[-1, 0, 1], average=None
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

    # Classification report
    class_report = classification_report(
        y_true, y_pred, labels=[-1, 0, 1],
        target_names=['negative', 'neutral', 'positive'],
        output_dict=True
    )

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
        'classification_report': class_report
    }

    logger.info(f"✓ Accuracy: {accuracy:.4f}")
    logger.info(f"✓ Weighted F1: {weighted_f1:.4f}")
    logger.info(f"✓ Macro F1: {macro_f1:.4f}")

    return metrics, y_pred, y_true

# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_results(metrics: Dict, y_true: np.ndarray, y_pred: np.ndarray):
    """Create visualization plots."""
    logger.info("Creating visualization plots...")

    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix - Llama 3 QLoRA')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrix.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Classification report
    report = metrics['classification_report']
    classes = ['negative', 'neutral', 'positive']
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report - Llama 3 QLoRA')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'classification_report.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Plots saved to {PLOTS_DIR}")

# =============================================================================
# COMPARISON
# =============================================================================


def compare_with_distilroberta(llama_metrics: Dict):
    """Compare Llama 3 QLoRA with DistilRoBERTa."""
    logger.info("=" * 80)
    logger.info("COMPARISON WITH DISTILROBERTA")
    logger.info("=" * 80)

    # Load DistilRoBERTa metrics
    distilroberta_path = Path(
        '../phase2/distilroberta_absa_results/distilroberta_metrics.json')

    if distilroberta_path.exists():
        with open(distilroberta_path, 'r') as f:
            distilroberta_metrics = json.load(f)

        logger.info("\nModel Comparison:")
        logger.info("-" * 80)
        logger.info(
            f"{'Metric':<30} {'DistilRoBERTa':<20} {'Llama 3 QLoRA':<20}")
        logger.info("-" * 80)
        logger.info(
            f"{'Accuracy':<30} {distilroberta_metrics['accuracy']:<20.4f} {llama_metrics['accuracy']:<20.4f}")
        logger.info(
            f"{'Weighted F1':<30} {distilroberta_metrics['weighted_f1']:<20.4f} {llama_metrics['weighted_f1']:<20.4f}")
        logger.info(
            f"{'Macro F1':<30} {distilroberta_metrics['macro_f1']:<20.4f} {llama_metrics['macro_f1']:<20.4f}")
        logger.info("-" * 80)

        # Improvement
        acc_improvement = llama_metrics['accuracy'] - \
            distilroberta_metrics['accuracy']
        f1_improvement = llama_metrics['weighted_f1'] - \
            distilroberta_metrics['weighted_f1']

        logger.info(f"\nImprovement:")
        logger.info(
            f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
        logger.info(
            f"  Weighted F1: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
    else:
        logger.warning(
            f"DistilRoBERTa metrics not found at {distilroberta_path}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Main function to run QLoRA fine-tuning pipeline."""
    logger.info("=" * 80)
    logger.info("LLAMA 3 8B QLORA FINE-TUNING FOR ABSA")
    logger.info("=" * 80)

    # Configure GPU
    gpu_available, device = configure_gpu()

    # Load data
    train_df, val_df, test_df, y_train, y_val, y_test = load_and_prepare_data(
        data_file='absa_dataset.csv'
    )

    # Load model and tokenizer
    model, tokenizer = load_llama3_model_and_tokenizer()

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ABSADataset(
        train_df['text_for_transformer'].tolist(),
        y_train,
        tokenizer
    )
    val_dataset = ABSADataset(
        val_df['text_for_transformer'].tolist(),
        y_val,
        tokenizer
    )
    test_dataset = ABSADataset(
        test_df['text_for_transformer'].tolist(),
        y_test,
        tokenizer
    )

    # Train model
    trainer, train_metrics = train_llama3_qlora(
        model, tokenizer, train_dataset, val_dataset, train_df, val_df
    )

    # Use the best model from trainer (already loaded due to load_best_model_at_end=True)
    logger.info("Using best model for evaluation...")
    best_model = trainer.model

    # Evaluate
    metrics, y_pred, y_true = evaluate_llama3_qlora(
        best_model, tokenizer, test_dataset, y_test
    )

    # Create visualizations
    plot_results(metrics, y_true, y_pred)

    # Compare with DistilRoBERTa
    compare_with_distilroberta(metrics)

    # Save metrics
    metrics_path = OUTPUT_DIR / 'llama3_qlora_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved: {metrics_path}")

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
