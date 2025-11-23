"""
Threshold-Optimized Training Script
Find optimal decision threshold for best balanced accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import json
import matplotlib.pyplot as plt

class SimpleDeepfakeDataset(Dataset):
    """Dataset for handcrafted features"""
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features
        features = sample['features']
        feature_values = np.array([v for v in features.values()], dtype=np.float32)
        
        # Handle inf/nan
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=999.0, neginf=-999.0)
        
        label = sample['label']
        
        return torch.FloatTensor(feature_values), torch.LongTensor([label])[0]


class OptimizedDeepfakeDetector(nn.Module):
    """Optimized deep neural network"""
    
    def __init__(self, input_dim=45, dropout=0.4):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 4
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 5
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 6
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            
            # Output
            nn.Linear(64, 2)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model and return probabilities"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_labels), np.array(all_probs), np.array(all_preds)


def find_optimal_threshold(labels, probs):
    """Find optimal threshold that maximizes balanced accuracy"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    # Balanced accuracy = (TPR + TNR) / 2 = (Sensitivity + Specificity) / 2
    tnr = 1 - fpr
    balanced_acc = (tpr + tnr) / 2
    
    # Find threshold with best balanced accuracy
    best_idx = np.argmax(balanced_acc)
    best_threshold = thresholds[best_idx]
    best_balanced_acc = balanced_acc[best_idx]
    
    return best_threshold, best_balanced_acc


def evaluate_with_threshold(labels, probs, threshold):
    """Evaluate with custom threshold"""
    preds = (probs >= threshold).astype(int)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
    cm = confusion_matrix(labels, preds)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'balanced_accuracy': balanced_acc
    }


def main():
    print("\n" + "="*70)
    print("THRESHOLD-OPTIMIZED DEEPFAKE DETECTOR - TRAINING")
    print("="*70 + "\n")
    
    # Config
    data_path = "outputs/features/dataset_features_chrom.pkl"
    batch_size = 64
    epochs = 100
    learning_rate = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Device: {device}")
    print(f"📦 Loading data from: {data_path}\n")
    
    # Load data
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"✅ Loaded {len(all_data)} samples")
    
    # Count classes
    labels = [d['label'] for d in all_data]
    real_count = labels.count(0)
    fake_count = labels.count(1)
    print(f"   Real: {real_count} ({real_count/len(labels)*100:.1f}%)")
    print(f"   Fake: {fake_count} ({fake_count/len(labels)*100:.1f}%)")
    
    # Get feature dimension
    feature_dim = len(all_data[0]['features'])
    print(f"   Features: {feature_dim}\n")
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Create datasets
    train_dataset = SimpleDeepfakeDataset(train_data)
    val_dataset = SimpleDeepfakeDataset(val_data)
    test_dataset = SimpleDeepfakeDataset(test_data)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset splits:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples\n")
    
    # Create model
    model = OptimizedDeepfakeDetector(input_dim=feature_dim, dropout=0.4).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Model: Deep Neural Network (6 layers)")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: ~{total_params*4/1024/1024:.2f} MB\n")
    
    # Class weights for imbalanced data
    class_weights = torch.FloatTensor([fake_count / real_count, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    best_balanced_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_labels, val_probs, val_preds = evaluate(model, val_loader, criterion, device)
        
        # Find optimal threshold
        optimal_threshold, balanced_acc = find_optimal_threshold(val_labels, val_probs)
        
        # Evaluate with optimal threshold
        metrics = evaluate_with_threshold(val_labels, val_probs, optimal_threshold)
        
        # Update scheduler
        scheduler.step(balanced_acc)
        
        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   Balanced Acc: {balanced_acc*100:.2f}% (Sensitivity: {metrics['sensitivity']:.4f}, Specificity: {metrics['specificity']:.4f})")
        print(f"   Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
        
        # Save best model
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_epoch = epoch
            patience_counter = 0
            
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'balanced_acc': balanced_acc,
                'optimal_threshold': optimal_threshold,
                'metrics': metrics
            }, 'checkpoints/best_model_optimized.pth')
            
            print(f"   ✅ New best model saved! (Balanced Acc: {balanced_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✅ Best model: Epoch {best_epoch} with Balanced Acc: {best_balanced_acc*100:.2f}%\n")
    
    # Load best model and evaluate on test set
    print("📊 Evaluating on test set...\n")
    checkpoint = torch.load('checkpoints/best_model_optimized.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold = checkpoint['optimal_threshold']
    
    test_loss, test_labels, test_probs, test_preds = evaluate(model, test_loader, criterion, device)
    
    # Evaluate with default threshold (0.5)
    default_metrics = evaluate_with_threshold(test_labels, test_probs, 0.5)
    
    # Evaluate with optimal threshold
    optimal_metrics = evaluate_with_threshold(test_labels, test_probs, optimal_threshold)
    
    print("="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    
    print(f"\n📊 With Default Threshold (0.5):")
    print(f"   Accuracy:  {default_metrics['accuracy']*100:.2f}%")
    print(f"   F1-Score:  {default_metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {default_metrics['auc']:.4f}")
    print(f"   Sensitivity: {default_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {default_metrics['specificity']:.4f}")
    print(f"   Balanced Acc: {default_metrics['balanced_accuracy']*100:.2f}%")
    
    cm = default_metrics['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"   FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    
    print(f"\n📊 With Optimal Threshold ({optimal_threshold:.4f}):")
    print(f"   Accuracy:  {optimal_metrics['accuracy']*100:.2f}%")
    print(f"   F1-Score:  {optimal_metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {optimal_metrics['auc']:.4f}")
    print(f"   Sensitivity: {optimal_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {optimal_metrics['specificity']:.4f}")
    print(f"   Balanced Acc: {optimal_metrics['balanced_accuracy']*100:.2f}%")
    
    cm = optimal_metrics['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"   FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    
    # Save results
    results = {
        'default_threshold': {
            'threshold': 0.5,
            'metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                       for k, v in default_metrics.items()}
        },
        'optimal_threshold': {
            'threshold': float(optimal_threshold),
            'metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                       for k, v in optimal_metrics.items()}
        }
    }
    
    Path("outputs").mkdir(exist_ok=True)
    with open('outputs/test_results_optimized.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: outputs/test_results_optimized.json")
    print(f"✅ Model saved to: checkpoints/best_model_optimized.pth")
    
    # Comparison
    print(f"\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<20} {'Default (0.5)':<20} {'Optimized':<20} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in ['accuracy', 'f1', 'balanced_accuracy', 'specificity', 'sensitivity']:
        default_val = default_metrics[metric]
        optimal_val = optimal_metrics[metric]
        improvement = optimal_val - default_val
        
        print(f"{metric.replace('_', ' ').title():<20} {default_val:>6.4f}             {optimal_val:>6.4f}             {improvement:>+6.4f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

