"""
Final Advanced Training Script for Deepfake Detection
Multiple improvements without reprocessing data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import json

class SimpleDeepfakeDataset(Dataset):
    """Dataset with better augmentation"""
    
    def __init__(self, data, use_augment=False):
        self.data = data
        self.use_augment = use_augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features
        features = sample['features']
        feature_values = np.array([v for v in features.values()], dtype=np.float32)
        
        # Handle inf/nan
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=999.0, neginf=-999.0)
        
        # Advanced augmentation
        if self.use_augment:
            # Gaussian noise
            noise = np.random.normal(0, 0.01, feature_values.shape)
            feature_values = feature_values + noise
            
            # Random scaling
            scale = np.random.uniform(0.95, 1.05)
            feature_values = feature_values * scale
        
        label = sample['label']
        
        return torch.FloatTensor(feature_values), torch.LongTensor([label])[0]


class EnsembleDeepfakeDetector(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, input_dim=45, dropout=0.3):
        super().__init__()
        
        # Model 1: Deep MLP
        self.model1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 2)
        )
        
        # Model 2: Wide MLP
        self.model2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 2)
        )
        
        # Model 3: Residual connections
        self.input_proj = nn.Linear(input_dim, 128)
        self.res1 = self._make_residual_block(128, 128, dropout)
        self.res2 = self._make_residual_block(128, 128, dropout)
        self.res3 = self._make_residual_block(128, 64, dropout)
        self.output3 = nn.Linear(64, 2)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.apply(self._init_weights)
    
    def _make_residual_block(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Model 1
        out1 = self.model1(x)
        
        # Model 2
        out2 = self.model2(x)
        
        # Model 3 with residuals
        x3 = self.input_proj(x)
        x3 = x3 + self.res1(x3)
        x3 = x3 + self.res2(x3)
        x3 = self.res3(x3)
        out3 = self.output3(x3)
        
        # Weighted ensemble
        weights = torch.softmax(self.ensemble_weights, dim=0)
        ensemble_out = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
        
        return ensemble_out


class ImbalancedFocalLoss(nn.Module):
    """Focal Loss with class balancing"""
    
    def __init__(self, alpha=0.1, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Higher alpha for minority class (real videos)
        alpha_t = torch.where(targets == 0, self.alpha * 10, self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        # Mixup for minority class
        if use_mixup and np.random.rand() < 0.4:
            alpha = 0.3
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(features.size(0)).to(device)
            
            mixed_features = lam * features + (1 - lam) * features[index]
            labels_a, labels_b = labels, labels[index]
            
            optimizer.zero_grad()
            outputs = model(mixed_features)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
    """Evaluate the model"""
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
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, precision, recall, f1, auc, cm


def main():
    print("\n" + "="*70)
    print("FINAL OPTIMIZED DEEPFAKE DETECTOR - TRAINING")
    print("="*70 + "\n")
    
    # Config
    data_path = "outputs/features/dataset_features_chrom.pkl"
    batch_size = 32  # Smaller batch size for better generalization
    epochs = 200
    learning_rate = 0.0003
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
    
    # Stratified split to ensure balanced validation/test sets
    from sklearn.model_selection import train_test_split
    
    # First split: 70% train, 30% temp
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Second split: 15% val, 15% test
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Oversample minority class in training
    train_real = [d for d in train_data if d['label'] == 0]
    train_fake = [d for d in train_data if d['label'] == 1]
    
    # Replicate real videos 5x
    train_real_oversampled = train_real * 5
    train_data_balanced = train_real_oversampled + train_fake
    
    np.random.shuffle(train_data_balanced)
    
    print(f"📊 Original training: {len(train_data)} samples")
    print(f"📊 Balanced training: {len(train_data_balanced)} samples")
    print(f"   (Real oversampled 5x: {len(train_real_oversampled)} real, {len(train_fake)} fake)\n")
    
    # Create datasets
    train_dataset = SimpleDeepfakeDataset(train_data_balanced, use_augment=True)
    val_dataset = SimpleDeepfakeDataset(val_data)
    test_dataset = SimpleDeepfakeDataset(test_data)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset splits:")
    print(f"   Train: {len(train_dataset)} samples (balanced)")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples\n")
    
    # Create model
    model = EnsembleDeepfakeDetector(input_dim=feature_dim, dropout=0.3).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Model: Ensemble (3 models)")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: ~{total_params*4/1024/1024:.2f} MB\n")
    
    # Loss and optimizer
    criterion = ImbalancedFocalLoss(alpha=0.1, gamma=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_cm = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Calculate specificity and sensitivity
        if val_cm.size > 0:
            tn = val_cm[0,0] if val_cm.shape[0] > 0 and val_cm.shape[1] > 0 else 0
            fp = val_cm[0,1] if val_cm.shape[0] > 0 and val_cm.shape[1] > 1 else 0
            fn = val_cm[1,0] if val_cm.shape[0] > 1 and val_cm.shape[1] > 0 else 0
            tp = val_cm[1,1] if val_cm.shape[0] > 1 and val_cm.shape[1] > 1 else 0
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            specificity = sensitivity = 0
        
        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"   Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"   Val AUC-ROC: {val_auc:.4f} | Specificity: {specificity:.4f} | Sensitivity: {sensitivity:.4f}")
        
        # Save best model based on F1 score (better metric for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
            }, 'checkpoints/best_model_final.pth')
            
            print(f"   ✅ New best model saved! (F1: {val_f1:.4f})")
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
    print(f"\n✅ Best model: Epoch {best_epoch} with Val F1: {best_val_f1:.4f}\n")
    
    # Load best model and evaluate on test set
    print("📊 Evaluating on test set...\n")
    checkpoint = torch.load('checkpoints/best_model_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_cm = evaluate(
        model, test_loader, criterion, device
    )
    
    print("="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {test_acc*100:.2f}%")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1-Score:  {test_f1:.4f}")
    print(f"   AUC-ROC:   {test_auc:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(f"   TN: {test_cm[0,0]:4d}  |  FP: {test_cm[0,1]:4d}")
    print(f"   FN: {test_cm[1,0]:4d}  |  TP: {test_cm[1,1]:4d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = test_cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n   Sensitivity (TPR): {sensitivity:.4f}")
    print(f"   Specificity (TNR): {specificity:.4f}")
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'confusion_matrix': test_cm.tolist(),
        'best_epoch': int(best_epoch),
        'best_val_f1': float(best_val_f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }
    
    with open('outputs/test_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: outputs/test_results_final.json")
    print(f"✅ Model saved to: checkpoints/best_model_final.pth")
    
    # Compare with previous models
    print(f"\n" + "="*70)
    print("COMPARISON: All Models")
    print("="*70)
    print(f"\n{'Metric':<20} {'Simple':<15} {'Enhanced':<15} {'Final':<15}")
    print("-" * 70)
    
    try:
        with open('outputs/test_results.json', 'r') as f:
            simple_results = json.load(f)
        with open('outputs/test_results_enhanced.json', 'r') as f:
            enhanced_results = json.load(f)
        
        print(f"{'Accuracy':<20} {simple_results['test_accuracy']*100:>6.2f}%        {enhanced_results['test_accuracy']*100:>6.2f}%        {test_acc*100:>6.2f}%")
        print(f"{'F1-Score':<20} {simple_results['test_f1']:>6.4f}          {enhanced_results['test_f1']:>6.4f}          {test_f1:>6.4f}")
        print(f"{'AUC-ROC':<20} {simple_results['test_auc']:>6.4f}          {enhanced_results['test_auc']:>6.4f}          {test_auc:>6.4f}")
        print(f"{'Specificity':<20} {0.0:>6.4f}          {enhanced_results['specificity']:>6.4f}          {specificity:>6.4f}")
        print(f"{'Sensitivity':<20} {1.0:>6.4f}          {enhanced_results['sensitivity']:>6.4f}          {sensitivity:>6.4f}")
    except:
        pass
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

