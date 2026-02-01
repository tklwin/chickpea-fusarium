"""
Research-Grade Evaluation Metrics for Thesis Experiments.

Includes:
- Standard metrics: Accuracy, Precision, Recall, F1
- Advanced metrics: Cohen's Kappa, MCC, Balanced Accuracy, AUC-ROC
- Per-class breakdown for imbalanced dataset analysis
- Confusion matrix visualization
- Statistical significance testing utilities
"""

from typing import Dict, List, Optional, Tuple, Union
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    auc,
)


# Class name mapping for reports
CLASS_NAMES = ["Resistant", "Moderate", "Susceptible"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Compute comprehensive research-grade classification metrics.
    
    Metrics included:
    - Accuracy (overall)
    - Balanced Accuracy (accounts for class imbalance)
    - Precision, Recall, F1 (weighted & macro)
    - Cohen's Kappa (agreement beyond chance)
    - Matthews Correlation Coefficient (MCC) - robust for imbalanced data
    - Per-class Precision, Recall, F1, Accuracy
    - AUC-ROC (if probabilities provided)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC-ROC)
        num_classes: Number of classes
    
    Returns:
        Dict with all computed metrics
    """
    metrics = {}
    
    # ========== Overall Metrics ==========
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    # Weighted averages (accounts for class frequency)
    metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Macro averages (treats all classes equally - important for imbalanced data)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # ========== Advanced Metrics (Research-Grade) ==========
    # Cohen's Kappa: Agreement beyond chance (-1 to 1, >0.8 is excellent)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient: Robust for imbalanced data (-1 to 1)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    
    # ========== Per-Class Metrics ==========
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i in range(num_classes):
        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        metrics[f"precision_{class_name}"] = per_class_precision[i] if i < len(per_class_precision) else 0
        metrics[f"recall_{class_name}"] = per_class_recall[i] if i < len(per_class_recall) else 0
        metrics[f"f1_{class_name}"] = per_class_f1[i] if i < len(per_class_f1) else 0
        
        # Per-class accuracy (specificity for that class)
        mask = y_true == i
        if mask.sum() > 0:
            metrics[f"accuracy_{class_name}"] = (y_pred[mask] == i).mean()
        else:
            metrics[f"accuracy_{class_name}"] = 0.0
    
    # ========== AUC-ROC (if probabilities provided) ==========
    if y_prob is not None:
        try:
            # Multi-class AUC (One-vs-Rest)
            metrics["auc_ovr_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
            metrics["auc_ovr_weighted"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
            
            # Per-class AUC
            for i in range(num_classes):
                class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) > 1:  # Need both classes present
                    metrics[f"auc_{class_name}"] = roc_auc_score(y_true_binary, y_prob[:, i])
        except Exception as e:
            # AUC computation can fail with certain class distributions
            pass
    
    # ========== Legacy keys for backward compatibility ==========
    metrics["precision"] = metrics["precision_weighted"]
    metrics["recall"] = metrics["recall_weighted"]
    metrics["f1"] = metrics["f1_weighted"]
    
    return metrics


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics with bootstrap confidence intervals.
    
    Useful for reporting statistical significance in thesis.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 95%)
    
    Returns:
        Dict with mean, lower, upper bounds for each metric
    """
    n_samples = len(y_true)
    metrics_list = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metrics
        metrics_list.append(compute_metrics(y_true_boot, y_pred_boot))
    
    # Aggregate results
    result = {}
    alpha = 1 - confidence
    
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        result[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "lower": np.percentile(values, alpha / 2 * 100),
            "upper": np.percentile(values, (1 - alpha / 2) * 100),
        }
    
    return result


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Get detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
    
    Returns:
        Formatted classification report string
    """
    if class_names is None:
        class_names = ["Resistant", "Moderate", "Susceptible"]
    
    return classification_report(y_true, y_pred, target_names=class_names)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: Whether to normalize the matrix
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ["Resistant", "Moderate", "Susceptible"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves (loss and accuracy).
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Training accuracy per epoch
        val_accs: Validation accuracy per epoch
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train')
    axes[1].plot(epochs, val_accs, 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add target line at 95%
    axes[1].axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Target (95%)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    return fig


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> Dict[int, float]:
    """
    Compute accuracy for each class separately.
    
    Useful for checking if model is biased towards majority class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dict mapping class index to accuracy
    """
    per_class_acc = {}
    
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[c] = (y_pred[mask] == c).mean()
        else:
            per_class_acc[c] = 0.0
    
    return per_class_acc
