"""
=============================================================================
Baseline Model for Aspect-Based Sentiment Analysis (ABSA)
=============================================================================

This script implements a TF-IDF + Logistic Regression pipeline for ABSA
on the preprocessed entity-aware dataset.

Features:
- TF-IDF vectorization with bigrams
- Logistic Regression with L1 regularization
- Train/Validation/Test split (80/10/10)
- Comprehensive evaluation metrics
- Feature importance analysis
- Multiple visualizations

Author: NLP Engineering Team
Date: 2024
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from typing import Dict, Tuple, List
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

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(file_path: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load the preprocessed ABSA dataset and extract features and labels.
    
    Args:
        file_path: Path to the CSV file containing preprocessed ABSA data
        
    Returns:
        Tuple of (X, y) where X is the clean_text series and y is sentiment labels
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    logger.info(f"Loading data from {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Extract features and labels
    X = df['clean_text'].copy()
    y = df['sentiment'].copy()
    
    # Handle empty text inputs
    empty_mask = (X.isna()) | (X.str.strip() == '')
    if empty_mask.sum() > 0:
        logger.warning(f"Found {empty_mask.sum()} empty text inputs. Filling with 'unknown'.")
        X[empty_mask] = 'unknown'
    
    logger.info(f"✓ Loaded {len(X)} samples")
    logger.info(f"Sentiment distribution:\n{y.value_counts().sort_index()}")
    
    return X, y


# =============================================================================
# DATA SPLITTING
# =============================================================================

def split_data(X: pd.Series, y: pd.Series, 
               train_size: float = 0.8, val_size: float = 0.1, 
               test_size: float = 0.1, random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Labels
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("\n" + "=" * 80)
    logger.info("SPLITTING DATA")
    logger.info("=" * 80)
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    
    logger.info(f"Training set size: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"Validation set size: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"Test set size: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# TF-IDF VECTORIZATION
# =============================================================================

def create_tfidf_vectorizer(max_features: int = 5000, 
                            ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """
    Create and configure TF-IDF vectorizer.
    
    Args:
        max_features: Maximum number of features to keep
        ngram_range: Range of n-grams to extract
        
    Returns:
        Configured TfidfVectorizer instance
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words='english',
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    return vectorizer


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, 
                              C: float = 0.1, max_iter: int = 1000) -> LogisticRegression:
    """
    Train a Logistic Regression model with specified parameters.
    
    Args:
        X_train: Training features (TF-IDF vectors)
        y_train: Training labels
        C: Inverse of regularization strength (smaller = stronger regularization)
        max_iter: Maximum number of iterations
        
    Returns:
        Trained LogisticRegression model
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING LOGISTIC REGRESSION MODEL")
    logger.info("=" * 80)
    logger.info(f"Parameters: C={C}, max_iter={max_iter}")
    
    model = LogisticRegression(
        solver='saga',
        max_iter=max_iter,
        class_weight='balanced',
        C=C,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("✓ Model training completed!")
    
    return model


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model: LogisticRegression, X: np.ndarray, y: np.ndarray, 
                   set_name: str = "Test") -> Dict:
    """
    Evaluate the model and compute various metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        set_name: Name of the dataset (for logging)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info(f"\nEvaluating on {set_name} set...")
    
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
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
    n_classes = 3
    
    # Calculate ROC curve and AUC for each class
    roc_auc_scores = {}
    roc_curves = {}
    
    for i, class_label in enumerate([-1, 0, 1]):
        if len(np.unique(y_binarized[:, i])) > 1:  # Check if class exists
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
    plt.title('Confusion Matrix - ABSA Baseline Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Confusion matrix saved to {save_path}")


def plot_classification_report_table(metrics: Dict, save_path: str) -> None:
    """Plot classification report as a table."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
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
    
    # Add weighted average
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
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(data) + 1):
        for j in range(5):
            if i == len(data):  # Last row (weighted avg)
                table[(i, j)].set_facecolor('#E8F5E9')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Classification Report - ABSA Baseline Model', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Classification report saved to {save_path}")


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
    plt.title('ROC Curves (One-vs-Rest) - ABSA Baseline Model', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ ROC curves saved to {save_path}")


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(model: LogisticRegression, vectorizer: TfidfVectorizer, 
                               top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """Analyze feature importance by extracting top positive and negative words for each class."""
    logger.info(f"\nAnalyzing top {top_n} features for each class...")
    
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = {}
    
    # Get coefficients for each class
    class_labels = ['Negative', 'Neutral', 'Positive']
    class_values = [-1, 0, 1]
    
    for class_val, class_label in zip(class_values, class_labels):
        # Get coefficients for this class
        coef_idx = list(model.classes_).index(class_val)
        coefficients = model.coef_[coef_idx]
        
        # Get top positive and negative features
        top_positive_indices = np.argsort(coefficients)[-top_n:][::-1]
        top_negative_indices = np.argsort(coefficients)[:top_n]
        
        top_positive = [(feature_names[i], float(coefficients[i])) for i in top_positive_indices]
        top_negative = [(feature_names[i], float(coefficients[i])) for i in top_negative_indices]
        
        feature_importance[class_label] = {
            'top_positive': top_positive,
            'top_negative': top_negative
        }
    
    return feature_importance


def plot_feature_importance(feature_importance: Dict, save_path: str, top_n: int = 20) -> None:
    """Plot feature importance for each class."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    class_labels = ['Negative', 'Neutral', 'Positive']
    
    for idx, class_label in enumerate(class_labels):
        # Top positive features
        pos_features = feature_importance[class_label]['top_positive'][:top_n]
        words = [f[0] for f in pos_features]
        scores = [f[1] for f in pos_features]
        
        axes[idx, 0].barh(range(len(words)), scores, color='green', alpha=0.7)
        axes[idx, 0].set_yticks(range(len(words)))
        axes[idx, 0].set_yticklabels(words)
        axes[idx, 0].set_xlabel('Coefficient Value', fontsize=10)
        axes[idx, 0].set_title(f'{class_label} - Top {top_n} Positive Features', fontsize=12, fontweight='bold')
        axes[idx, 0].invert_yaxis()
        axes[idx, 0].grid(True, alpha=0.3, axis='x')
        
        # Top negative features
        neg_features = feature_importance[class_label]['top_negative'][:top_n]
        words = [f[0] for f in neg_features]
        scores = [f[1] for f in neg_features]
        
        axes[idx, 1].barh(range(len(words)), scores, color='red', alpha=0.7)
        axes[idx, 1].set_yticks(range(len(words)))
        axes[idx, 1].set_yticklabels(words)
        axes[idx, 1].set_xlabel('Coefficient Value', fontsize=10)
        axes[idx, 1].set_title(f'{class_label} - Top {top_n} Negative Features', fontsize=12, fontweight='bold')
        axes[idx, 1].invert_yaxis()
        axes[idx, 1].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Feature Importance - ABSA Baseline Model', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Feature importance plot saved to {save_path}")


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def perform_cross_validation(model: LogisticRegression, X_train: np.ndarray, 
                            y_train: np.ndarray, cv: int = 5) -> Dict:
    """Perform k-fold cross-validation and return results."""
    logger.info(f"\nPerforming {cv}-fold cross-validation...")
    
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    
    cv_results = {
        'mean_accuracy': float(np.mean(cv_scores)),
        'std_accuracy': float(np.std(cv_scores)),
        'fold_scores': [float(score) for score in cv_scores]
    }
    
    logger.info(f"CV Mean Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    
    return cv_results


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(model: LogisticRegression, vectorizer: TfidfVectorizer, 
                metrics: Dict, feature_importance: Dict, cv_results: Dict,
                model_path: str, metrics_path: str) -> None:
    """Save model, vectorizer, and evaluation metrics."""
    # Save model and vectorizer
    logger.info(f"\nSaving model to {model_path}...")
    joblib.dump({
        'model': model,
        'vectorizer': vectorizer
    }, model_path)
    logger.info("✓ Model saved successfully!")
    
    # Save metrics
    logger.info(f"Saving metrics to {metrics_path}...")
    all_metrics = {
        'test_metrics': metrics,
        'cross_validation': cv_results,
        'feature_importance': feature_importance
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("✓ Metrics saved successfully!")


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_metrics_summary(val_metrics: Dict, test_metrics: Dict, cv_results: Dict) -> None:
    """Print a summary of evaluation metrics."""
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("="*80)
    
    logger.info("\nValidation Set:")
    logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"  Weighted F1-Score: {val_metrics['weighted_f1']:.4f}")
    
    logger.info("\nTest Set:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
    logger.info(f"  Macro F1-Score: {test_metrics['macro_f1']:.4f}")
    
    logger.info(f"\nCross-Validation Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    
    logger.info("\nPer-Class Metrics (Test Set):")
    logger.info("-" * 80)
    for label, label_name in [('negative', 'Negative'), ('neutral', 'Neutral'), ('positive', 'Positive')]:
        pc = test_metrics['per_class_metrics'][label]
        logger.info(f"{label_name:10s} | Precision: {pc['precision']:.4f} | Recall: {pc['recall']:.4f} | F1: {pc['f1']:.4f} | Support: {pc['support']}")
    
    logger.info("\nROC AUC Scores (Test Set):")
    for class_key, auc_score in test_metrics['roc_auc_scores'].items():
        class_name = class_key.replace('class_', '').replace('-1', 'Negative').replace('0', 'Neutral').replace('1', 'Positive')
        logger.info(f"  {class_name}: {auc_score:.4f}")
    
    logger.info("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run the baseline ABSA model pipeline."""
    # Configuration
    DATA_FILE = 'absa_dataset.csv'
    OUTPUT_DIR = Path('baseline_absa_results')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    MODEL_PATH = OUTPUT_DIR / 'baseline_absa_model.joblib'
    METRICS_PATH = OUTPUT_DIR / 'baseline_absa_metrics.json'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_data(DATA_FILE)
    
    # Split data (80/10/10)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
    )
    
    # Create and fit TF-IDF vectorizer
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TF-IDF VECTORIZER")
    logger.info("=" * 80)
    vectorizer = create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    logger.info(f"✓ TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # Train model
    model = train_logistic_regression(X_train_tfidf, y_train, C=0.1, max_iter=1000)
    
    # Cross-validation
    cv_results = perform_cross_validation(model, X_train_tfidf, y_train, cv=5)
    
    # Evaluate on validation set
    val_metrics, _, _, _ = evaluate_model(model, X_val_tfidf, y_val, set_name="Validation")
    
    # Evaluate on test set
    test_metrics, y_pred, y_pred_proba, roc_curves = evaluate_model(
        model, X_test_tfidf, y_test, set_name="Test"
    )
    
    # Print metrics summary
    print_metrics_summary(val_metrics, test_metrics, cv_results)
    
    # Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    plot_confusion_matrix(
        np.array(test_metrics['confusion_matrix']),
        str(PLOTS_DIR / 'confusion_matrix.png')
    )
    plot_classification_report_table(
        test_metrics,
        str(PLOTS_DIR / 'classification_report.png')
    )
    plot_roc_curves(
        roc_curves,
        test_metrics,
        str(PLOTS_DIR / 'roc_curves.png')
    )
    
    # Feature importance analysis
    feature_importance = analyze_feature_importance(model, vectorizer, top_n=20)
    plot_feature_importance(
        feature_importance,
        str(PLOTS_DIR / 'feature_importance.png'),
        top_n=20
    )
    
    # Save results
    save_results(
        model, vectorizer, test_metrics, feature_importance, cv_results,
        str(MODEL_PATH), str(METRICS_PATH)
    )
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE ABSA MODEL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nAll results saved to '{OUTPUT_DIR}/' directory")
    logger.info(f"Model saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()

