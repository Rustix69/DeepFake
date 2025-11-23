"""
Simple Training Script for Deepfake Detection
Uses only handcrafted features with a simple neural network
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
    """Simple dataset that uses only handcrafted features"""
    
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
        
        # Simple augmentation for training
        if self.use_augment:
            noise = np.random.normal(0, 0.01, feature_values.shape)
            feature_values = feature_values + noise
        
        # Label
        label = sample['label']
        
        return torch.FloatTensor(feature_values), torch.LongTensor([label])[0]


class SimpleDeepfakeDetector(nn.Module):
    """Simple MLP for deepfake detection"""
    
    def __init__(self, input_dim=45, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


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
    print("SIMPLE DEEPFAKE DETECTOR - TRAINING")
    print("="*70 + "\n")
    
    # Config
    data_path = "outputs/features/dataset_features_chrom.pkl"
    batch_size = 64
    epochs = 100
    learning_rate = 0.001
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Dataset splits:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples\n")
    
    # Create model
    model = SimpleDeepfakeDetector(input_dim=feature_dim).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Model: Simple MLP")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: ~{total_params*4/1024/1024:.2f} MB\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
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
            }, 'checkpoints/best_model_simple.pth')
            
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
    checkpoint = torch.load('checkpoints/best_model_simple.pth')
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
        'best_val_acc': float(best_val_acc)
    }
    
    with open('outputs/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: outputs/test_results.json")
    print(f"✅ Model saved to: checkpoints/best_model_simple.pth")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

