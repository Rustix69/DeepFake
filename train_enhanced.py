"""
Enhanced Training Script for Deepfake Detection
Uses advanced architecture with attention and better class balancing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import json

class SimpleDeepfakeDataset(Dataset):
    """Dataset for handcrafted features"""
    
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
        
        # Augmentation
        if self.use_augment:
            noise = np.random.normal(0, 0.02, feature_values.shape)
            feature_values = feature_values + noise
            
            # Random feature dropout (5% chance)
            if np.random.rand() < 0.05:
                mask = np.random.rand(len(feature_values)) > 0.1
                feature_values = feature_values * mask
        
        label = sample['label']
        
        return torch.FloatTensor(feature_values), torch.LongTensor([label])[0]


class AttentionBlock(nn.Module):
    """Self-attention for features"""
    
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        # x: (batch, dim)
        q = self.query(x).unsqueeze(1)  # (batch, 1, dim)
        k = self.key(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)
        
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).squeeze(1)
        
        return out + x  # Residual connection


class EnhancedDeepfakeDetector(nn.Module):
    """Enhanced MLP with attention and residual connections"""
    
    def __init__(self, input_dim=45, hidden_dims=[256, 256, 128, 128, 64], dropout=0.4):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention block
        self.attention = AttentionBlock(hidden_dims[0])
        
        # Deep layers with residual connections
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output head
        self.output = nn.Linear(hidden_dims[-1], 2)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.attention(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output(x)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=False):
    """Train for one epoch with optional mixup"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        # Mixup augmentation
        if use_mixup and np.random.rand() < 0.3:
            alpha = 0.2
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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, precision, recall, f1, auc, cm


def main():
    print("\n" + "="*70)
    print("ENHANCED DEEPFAKE DETECTOR - TRAINING")
    print("="*70 + "\n")
    
    # Config
    data_path = "outputs/features/dataset_features_chrom.pkl"
    batch_size = 64
    epochs = 150
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
    print(f"   Real: {labels.count(0)} | Fake: {labels.count(1)}")
    
    # Get feature dimension
    feature_dim = len(all_data[0]['features'])
    print(f"   Features: {feature_dim}\n")
    
    # Split data
    train_size = int(0.7 * len(all_data))
    val_size = int(0.15 * len(all_data))
    test_size = len(all_data) - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        all_data, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets
    train_dataset = SimpleDeepfakeDataset([all_data[i] for i in train_data.indices], use_augment=True)
    val_dataset = SimpleDeepfakeDataset([all_data[i] for i in val_data.indices])
    test_dataset = SimpleDeepfakeDataset([all_data[i] for i in test_data.indices])
    
    # Weighted sampler for class imbalance
    train_labels = [all_data[i]['label'] for i in train_data.indices]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset splits:")
    print(f"   Train: {len(train_dataset)} samples (weighted sampling)")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples\n")
    
    # Create model
    model = EnhancedDeepfakeDetector(input_dim=feature_dim).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Model: Enhanced MLP with Attention")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: ~{total_params*4/1024/1024:.2f} MB\n")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True, min_lr=1e-6
    )
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"   Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"   Val AUC-ROC: {val_auc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
            }, 'checkpoints/best_model_enhanced.pth')
            
            print(f"   ✅ New best model saved! (Acc: {val_acc*100:.2f}%)")
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
    print(f"\n✅ Best model: Epoch {best_epoch} with Val Acc: {best_val_acc*100:.2f}%\n")
    
    # Load best model and evaluate on test set
    print("📊 Evaluating on test set...\n")
    checkpoint = torch.load('checkpoints/best_model_enhanced.pth')
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
        'best_val_acc': float(best_val_acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }
    
    with open('outputs/test_results_enhanced.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: outputs/test_results_enhanced.json")
    print(f"✅ Model saved to: checkpoints/best_model_enhanced.pth")
    
    # Compare with simple model
    try:
        with open('outputs/test_results.json', 'r') as f:
            simple_results = json.load(f)
        
        print(f"\n" + "="*70)
        print("COMPARISON: Simple vs Enhanced Model")
        print("="*70)
        print(f"\n{'Metric':<20} {'Simple':<15} {'Enhanced':<15} {'Improvement':<15}")
        print("-" * 70)
        print(f"{'Accuracy':<20} {simple_results['test_accuracy']*100:>6.2f}%        {test_acc*100:>6.2f}%        {(test_acc - simple_results['test_accuracy'])*100:>+6.2f}%")
        print(f"{'AUC-ROC':<20} {simple_results['test_auc']:>6.4f}          {test_auc:>6.4f}          {(test_auc - simple_results['test_auc']):>+6.4f}")
        print(f"{'F1-Score':<20} {simple_results['test_f1']:>6.4f}          {test_f1:>6.4f}          {(test_f1 - simple_results['test_f1']):>+6.4f}")
        print(f"{'Specificity':<20} {0.0:>6.4f}          {specificity:>6.4f}          {specificity:>+6.4f}")
    except:
        pass
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

