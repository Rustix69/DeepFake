"""
Evaluation and Visualization for Deepfake Detection
Comprehensive metrics, confusion matrix, ROC curves, attention visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, average_precision_score, precision_recall_curve
)
import json


class DeepfakeEvaluator:
    """
    Comprehensive evaluation for deepfake detection model
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation on a dataset
        
        Args:
            dataloader: Data loader
            return_predictions: Whether to return full predictions
        
        Returns:
            Dictionary with metrics and optionally predictions
        """
        all_labels = []
        all_preds = []
        all_probs = []
        all_video_names = []
        
        print("\n🔍 Running evaluation...")
        for batch in tqdm(dataloader):
            rppg_signals = batch['rppg_signals'].to(self.device)
            labels = batch['labels'].to(self.device)
            handcrafted = batch.get('handcrafted_features')
            if handcrafted is not None:
                handcrafted = handcrafted.to(self.device)
            
            # Forward pass
            outputs = self.model(rppg_signals, handcrafted)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_probs.extend(outputs['probabilities'][:, 1].cpu().numpy())
            all_video_names.extend(batch['video_name'])
        
        # Convert to numpy
        labels = np.array(all_labels)
        predictions = np.array(all_preds)
        probabilities = np.array(all_probs)
        
        # Compute metrics
        metrics = self._compute_all_metrics(labels, predictions, probabilities)
        
        # Prepare output
        result = {'metrics': metrics}
        
        if return_predictions:
            result['predictions'] = {
                'labels': labels.tolist(),
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'video_names': all_video_names
            }
        
        return result
    
    def _compute_all_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """Compute comprehensive metrics"""
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # AUC metrics
        auc_roc = roc_auc_score(labels, probabilities)
        auc_pr = average_precision_score(labels, probabilities)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive/negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
        
        return metrics
    
    def visualize_results(
        self,
        evaluation_result: Dict,
        output_dir: str = "evaluation_results"
    ):
        """
        Create comprehensive visualizations
        
        Args:
            evaluation_result: Result from evaluate()
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics = evaluation_result['metrics']
        preds_data = evaluation_result.get('predictions')
        
        if preds_data is None:
            print("⚠️  No predictions data, skipping visualizations")
            return
        
        labels = np.array(preds_data['labels'])
        predictions = np.array(preds_data['predictions'])
        probabilities = np.array(preds_data['probabilities'])
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(labels, predictions, output_dir)
        
        # 2. ROC Curve
        self._plot_roc_curve(labels, probabilities, metrics['auc_roc'], output_dir)
        
        # 3. Precision-Recall Curve
        self._plot_pr_curve(labels, probabilities, metrics['auc_pr'], output_dir)
        
        # 4. Probability Distribution
        self._plot_probability_distribution(labels, probabilities, output_dir)
        
        # 5. Metrics Summary
        self._plot_metrics_summary(metrics, output_dir)
        
        print(f"\n💾 Visualizations saved to: {output_dir}")
    
    def _plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        output_dir: Path
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        auc_score: float,
        output_dir: Path
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        auc_pr: float,
        output_dir: Path
    ):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {auc_pr:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_probability_distribution(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        output_dir: Path
    ):
        """Plot probability distributions for real vs fake"""
        real_probs = probabilities[labels == 0]
        fake_probs = probabilities[labels == 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(real_probs, bins=20, alpha=0.6, label='Real', color='green', edgecolor='black')
        plt.hist(fake_probs, bins=20, alpha=0.6, label='Fake', color='red', edgecolor='black')
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
        plt.xlabel('Predicted Probability (Fake)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Probability Distribution: Real vs Fake', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_summary(
        self,
        metrics: Dict,
        output_dir: Path
    ):
        """Plot metrics summary"""
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['auc_roc'],
            metrics['specificity']
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(metric_names, metric_values, color='skyblue', edgecolor='navy')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(value + 0.01, i, f'{value:.4f}', va='center', fontweight='bold')
        
        ax.set_xlim([0, 1.1])
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Evaluation Metrics Summary', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(
        self,
        evaluation_result: Dict,
        output_path: str = "evaluation_report.json"
    ):
        """Save evaluation report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(evaluation_result, f, indent=2)
        print(f"📄 Evaluation report saved to: {output_path}")
    
    def print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy:    {metrics['accuracy']:.4f}")
        print(f"   Precision:   {metrics['precision']:.4f}")
        print(f"   Recall:      {metrics['recall']:.4f}")
        print(f"   F1-Score:    {metrics['f1_score']:.4f}")
        print(f"   AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"   AUC-PR:      {metrics['auc_pr']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Negatives:  {cm['tn']}")
        print(f"   False Positives: {cm['fp']}")
        print(f"   False Negatives: {cm['fn']}")
        print(f"   True Positives:  {cm['tp']}")
        
        print(f"\n🎯 Additional Metrics:")
        print(f"   Sensitivity:  {metrics['sensitivity']:.4f}")
        print(f"   Specificity:  {metrics['specificity']:.4f}")
        print(f"   FPR:          {metrics['fpr']:.4f}")
        print(f"   FNR:          {metrics['fnr']:.4f}")
        print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EVALUATOR MODULE - TESTING")
    print("="*70 + "\n")
    
    print("✅ Evaluator module loaded successfully")
    print("\nFeatures:")
    print("  • Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR)")
    print("  • Confusion matrix")
    print("  • ROC curve visualization")
    print("  • Precision-Recall curve")
    print("  • Probability distribution plots")
    print("  • Metrics summary visualization")
    print("  • JSON report export")
    print("\n" + "="*70 + "\n")

