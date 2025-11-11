"""
Model Evaluation Metrics for Price Direction Prediction

Implements comprehensive evaluation metrics for binary classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive evaluation for price direction prediction models.
    
    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - AUC-ROC
    - Confusion Matrix
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC-ROC)
            
        Returns:
            Dictionary of metrics
        """
        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # AUC-ROC (requires probabilities)
        if y_prob is not None:
            y_prob = y_prob.flatten()
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
                self.metrics['auc_roc'] = auc_roc
            except ValueError:
                self.metrics['auc_roc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm
        
        # Calculate additional metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            self.metrics['specificity'] = specificity
            
            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            self.metrics['false_positive_rate'] = fpr
            
            # False Negative Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            self.metrics['false_negative_rate'] = fnr
        
        return self.metrics
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print("\nClassification Metrics:")
        print("-"*60)
        for metric, value in self.metrics.items():
            if metric != 'confusion_matrix' and value is not None:
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        print("\n" + "-"*60)
        print("Detailed Classification Report:")
        print("-"*60)
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
        
        print("\nConfusion Matrix:")
        print("-"*60)
        cm = self.metrics.get('confusion_matrix')
        if cm is not None:
            print("              Predicted")
            print("              Down    Up")
            print(f"Actual Down   {cm[0][0]:4d}  {cm[0][1]:4d}")
            print(f"       Up     {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        print("="*60 + "\n")
    
    def plot_confusion_matrix(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        cm = self.metrics.get('confusion_matrix')
        if cm is None:
            print("No confusion matrix available")
            return
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Down', 'Up'],
            yticklabels=['Down', 'Up']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: str = None,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def calculate_directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy (trend prediction accuracy).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Directional accuracy
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        correct_direction = np.sum(y_true == y_pred)
        total = len(y_true)
        
        return correct_direction / total if total > 0 else 0.0
    
    def analyze_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: np.ndarray = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Analyze model performance across different classification thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: Array of thresholds to test
            
        Returns:
            Dictionary mapping thresholds to metrics
        """
        if thresholds is None:
            thresholds = np.arange(0.3, 0.8, 0.05)
        
        y_true = y_true.flatten()
        y_prob = y_prob.flatten()
        
        results = {}
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results[float(threshold)] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return results
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = 'f1_score'
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold based on a metric.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('accuracy', 'precision', 'recall', 'f1_score')
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        threshold_results = self.analyze_threshold(y_true, y_prob)
        
        best_threshold = 0.5
        best_value = 0.0
        
        for threshold, metrics in threshold_results.items():
            if metrics[metric] > best_value:
                best_value = metrics[metric]
                best_threshold = threshold
        
        return best_threshold, best_value
