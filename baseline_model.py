"""
Baseline Model for Financial Sentiment Analysis

This script implements a TF-IDF + Logistic Regression pipeline for financial
sentiment analysis on the preprocessed SEntFiN dataset.

Features:
- TF-IDF vectorization with bigrams
- Logistic Regression with L1 regularization
- 5-fold cross-validation
- Comprehensive evaluation metrics
- Feature importance analysis
- Multiple visualizations
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data(file_path: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load the preprocessed dataset and extract features and labels.
    
    Args:
        file_path: Path to the CSV file containing preprocessed data
        
    Returns:
        Tuple of (X, y) where X is the clean_text series and y is sentiment labels
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Extract features and labels
    X = df['clean_text'].copy()
    y = df['sentiment'].copy()
    
    # Handle empty text inputs
    empty_mask = (X.isna()) | (X.str.strip() == '')
    if empty_mask.sum() > 0:
        print(f"Warning: Found {empty_mask.sum()} empty text inputs. Filling with 'unknown'.")
        X[empty_mask] = 'unknown'
    
    print(f"Loaded {len(X)} samples")
    print(f"Sentiment distribution:\n{y.value_counts().sort_index()}")
    
    return X, y


def create_tfidf_vectorizer(max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
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
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        solver='saga',
        max_iter=max_iter,
        class_weight='balanced',
        C=C,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model


def evaluate_model(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate the model and compute various metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[-1, 0, 1], zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    
    # ROC AUC (one-vs-rest)
    y_test_binarized = label_binarize(y_test, classes=[-1, 0, 1])
    n_classes = 3
    
    # Calculate ROC curve and AUC for each class
    roc_auc_scores = {}
    roc_curves = {}
    
    for i, class_label in enumerate([-1, 0, 1]):
        if len(np.unique(y_test_binarized[:, i])) > 1:  # Check if class exists in test set
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
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
        'classification_report': classification_report(y_test, y_pred, labels=[-1, 0, 1], output_dict=True)
    }
    
    return metrics, y_pred, y_pred_proba, roc_curves


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """
    Plot and save confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix array
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_classification_report_table(metrics: Dict, save_path: str) -> None:
    """
    Plot classification report as a table.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_path: Path to save the plot
    """
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
    
    plt.title('Classification Report', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classification report saved to {save_path}")


def plot_roc_curves(roc_curves: Dict, metrics: Dict, save_path: str) -> None:
    """
    Plot ROC curves for each class (one-vs-rest).
    
    Args:
        roc_curves: Dictionary containing FPR and TPR for each class
        metrics: Dictionary containing ROC AUC scores
        save_path: Path to save the plot
    """
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
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {save_path}")


def plot_learning_curves(model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray,
                         vectorizer: TfidfVectorizer, save_path: str, cv: int = 5) -> None:
    """
    Plot learning curves showing training and validation accuracy.
    
    Args:
        model: Model to evaluate
        X_train: Training features
        y_train: Training labels
        vectorizer: Fitted vectorizer (for transforming data)
        save_path: Path to save the plot
        cv: Number of cross-validation folds
    """
    print("\nComputing learning curves...")
    
    # Use a wrapper that includes vectorization
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', model)
    ])
    
    # For learning curves, we need to work with raw text
    # So we'll use a simpler approach with already vectorized data
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Accuracy', linewidth=2)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Accuracy', linewidth=2)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Learning Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {save_path}")


def analyze_feature_importance(model: LogisticRegression, vectorizer: TfidfVectorizer, 
                               top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze feature importance by extracting top positive and negative words for each class.
    
    Args:
        model: Trained LogisticRegression model
        vectorizer: Fitted TfidfVectorizer
        top_n: Number of top features to extract per class
        
    Returns:
        Dictionary containing top features for each class
    """
    print(f"\nAnalyzing top {top_n} features for each class...")
    
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
    """
    Plot feature importance for each class.
    
    Args:
        feature_importance: Dictionary containing feature importance data
        save_path: Path to save the plot
        top_n: Number of top features to display
    """
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
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def perform_cross_validation(model: LogisticRegression, X_train: np.ndarray, 
                            y_train: np.ndarray, cv: int = 5) -> Dict:
    """
    Perform k-fold cross-validation and return results.
    
    Args:
        model: Model to evaluate
        X_train: Training features
        y_train: Training labels
        cv: Number of folds
        
    Returns:
        Dictionary containing CV results
    """
    print(f"\nPerforming {cv}-fold cross-validation...")
    
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
    
    print(f"CV Mean Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    
    return cv_results


def save_results(model: LogisticRegression, vectorizer: TfidfVectorizer, 
                metrics: Dict, feature_importance: Dict, cv_results: Dict,
                model_path: str, metrics_path: str) -> None:
    """
    Save model, vectorizer, and evaluation metrics.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        metrics: Evaluation metrics dictionary
        feature_importance: Feature importance dictionary
        cv_results: Cross-validation results
        model_path: Path to save the model
        metrics_path: Path to save the metrics
    """
    # Save model and vectorizer
    print(f"\nSaving model to {model_path}...")
    joblib.dump({
        'model': model,
        'vectorizer': vectorizer
    }, model_path)
    print("Model saved successfully!")
    
    # Save metrics
    print(f"Saving metrics to {metrics_path}...")
    all_metrics = {
        'test_metrics': metrics,
        'cross_validation': cv_results,
        'feature_importance': feature_importance
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved successfully!")


def print_metrics_summary(metrics: Dict, cv_results: Dict) -> None:
    """
    Print a summary of evaluation metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        cv_results: Cross-validation results
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"\nCross-Validation Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    for label, label_name in [('negative', 'Negative'), ('neutral', 'Neutral'), ('positive', 'Positive')]:
        pc = metrics['per_class_metrics'][label]
        print(f"{label_name:10s} | Precision: {pc['precision']:.4f} | Recall: {pc['recall']:.4f} | F1: {pc['f1']:.4f} | Support: {pc['support']}")
    
    print("\nROC AUC Scores:")
    for class_key, auc_score in metrics['roc_auc_scores'].items():
        class_name = class_key.replace('class_', '').replace('-1', 'Negative').replace('0', 'Neutral').replace('1', 'Positive')
        print(f"  {class_name}: {auc_score:.4f}")
    
    print("="*60)


def main():
    """Main function to run the baseline model pipeline."""
    # Configuration
    DATA_FILE = 'phase1_dataset.csv'
    OUTPUT_DIR = 'baseline_results'
    MODEL_PATH = 'baseline_model.joblib'
    METRICS_PATH = os.path.join(OUTPUT_DIR, 'baseline_metrics.json')
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    X, y = load_data(DATA_FILE)
    
    # Train-test split
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create and fit TF-IDF vectorizer
    print("\nCreating TF-IDF vectorizer...")
    vectorizer = create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # Train model
    model = train_logistic_regression(X_train_tfidf, y_train, C=0.1, max_iter=1000)
    
    # Cross-validation
    cv_results = perform_cross_validation(model, X_train_tfidf, y_train, cv=5)
    
    # Evaluate model
    metrics, y_pred, y_pred_proba, roc_curves = evaluate_model(model, X_test_tfidf, y_test)
    
    # Print metrics summary
    print_metrics_summary(metrics, cv_results)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    )
    plot_classification_report_table(
        metrics,
        os.path.join(OUTPUT_DIR, 'classification_report.png')
    )
    plot_roc_curves(
        roc_curves,
        metrics,
        os.path.join(OUTPUT_DIR, 'roc_curves.png')
    )
    plot_learning_curves(
        model,
        X_train_tfidf,
        y_train,
        vectorizer,
        os.path.join(OUTPUT_DIR, 'learning_curves.png'),
        cv=5
    )
    
    # Feature importance analysis
    feature_importance = analyze_feature_importance(model, vectorizer, top_n=20)
    plot_feature_importance(
        feature_importance,
        os.path.join(OUTPUT_DIR, 'feature_importance.png'),
        top_n=20
    )
    
    # Save results
    save_results(
        model, vectorizer, metrics, feature_importance, cv_results,
        MODEL_PATH, METRICS_PATH
    )
    
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll results saved to '{OUTPUT_DIR}/' directory")
    print(f"Model saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()

