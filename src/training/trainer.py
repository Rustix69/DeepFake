"""
Training Pipeline for Deepfake Detector
Production-quality training with all best practices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json
import time


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Labels (B,)
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Returns True if should stop training
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class DeepfakeTrainer:
    """
    Complete training pipeline for DeepfakeDetector
    
    Features:
    - Focal loss for class imbalance
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Mixed precision training
    - Checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        
        # Optimization
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        
        # Loss function
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        
        # Training params
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        use_amp: bool = True,  # Mixed precision
        
        # Scheduler
        use_scheduler: bool = True,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        
        # Early stopping
        early_stop_patience: int = 15,
        
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
        
        # Logging
        log_dir: str = "logs"
    ):
        """Initialize trainer"""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience,
                verbose=True
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stop_patience)
        
        # Training params
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Scaler for mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_val_loss = float('inf')
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Mixed precision: {self.use_amp}")
        print(f"   Loss: {'Focal' if use_focal_loss else 'CrossEntropy'}")
        print(f"   Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            rppg_signals = batch['rppg_signals'].to(self.device)
            labels = batch['labels'].to(self.device)
            handcrafted = batch.get('handcrafted_features')
            if handcrafted is not None:
                handcrafted = handcrafted.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(rppg_signals, handcrafted)
                    loss = self.criterion(outputs['logits'], labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(rppg_signals, handcrafted)
                loss = self.criterion(outputs['logits'], labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Optimizer step
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = outputs['predictions']
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, Dict]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.max_epochs} [Val]")
        
        for batch in pbar:
            rppg_signals = batch['rppg_signals'].to(self.device)
            labels = batch['labels'].to(self.device)
            handcrafted = batch.get('handcrafted_features')
            if handcrafted is not None:
                handcrafted = handcrafted.to(self.device)
            
            # Forward pass
            outputs = self.model(rppg_signals, handcrafted)
            loss = self.criterion(outputs['logits'], labels)
            
            # Statistics
            total_loss += loss.item()
            predictions = outputs['predictions']
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Store for metrics
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs['probabilities'][:, 1].cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Compute additional metrics
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        
        return avg_loss, accuracy, metrics
    
    def _compute_metrics(
        self,
        labels: List[int],
        predictions: List[int],
        probabilities: List[float]
    ) -> Dict:
        """Compute additional metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        labels = np.array(labels)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'auc': roc_auc_score(labels, probabilities),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        return metrics
    
    def train(self) -> Dict:
        """Complete training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Scheduler step
            if self.use_scheduler:
                self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.max_epochs}:")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"  Metrics: AUC={val_metrics['auc']:.4f}, F1={val_metrics['f1']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✅ New best model saved!")
            elif not self.save_best_only:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n⚠️  Early stopping triggered after epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
        
        # Save final history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.use_scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.use_scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 Training history saved to {history_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINER MODULE - TESTING")
    print("="*70 + "\n")
    
    print("✅ Trainer module loaded successfully")
    print("\nFeatures:")
    print("  • Focal Loss for class imbalance")
    print("  • AdamW optimizer with weight decay")
    print("  • Learning rate scheduling (ReduceLROnPlateau)")
    print("  • Early stopping")
    print("  • Gradient clipping")
    print("  • Mixed precision training (AMP)")
    print("  • Checkpointing (best & periodic)")
    print("  • TensorBoard logging")
    print("  • Comprehensive metrics (ACC, AUC, F1, etc.)")
    print("\n" + "="*70 + "\n")

