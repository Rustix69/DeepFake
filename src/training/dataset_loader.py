"""
Dataset Loader for rPPG-based Deepfake Detection
Loads preprocessed rPPG features and signals for model training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split


class RPPGDataset(Dataset):
    """
    Dataset for loading preprocessed rPPG signals and features
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        use_handcrafted: bool = True,
        sequence_length: int = 150,
        num_regions: int = 7,
        augment: bool = False
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to processed features pickle file
            split: 'train', 'val', or 'test'
            use_handcrafted: Whether to include hand-crafted features
            sequence_length: Expected sequence length
            num_regions: Number of ROIs
            augment: Whether to apply data augmentation (only for training)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.use_handcrafted = use_handcrafted
        self.sequence_length = sequence_length
        self.num_regions = num_regions
        self.augment = augment and (split == 'train')
        
        # Load data
        self._load_data()
        
        print(f"✅ Loaded {split} dataset: {len(self)} samples")
    
    def _load_data(self):
        """Load preprocessed data"""
        # Load pickle file with full results
        with open(self.data_path, 'rb') as f:
            all_results = pickle.load(f)
        
        # Check if dataset is large enough for stratified split
        labels = [r['label'] for r in all_results]
        min_class_count = min(labels.count(0), labels.count(1))
        
        # Need at least 6 per class for stratified 70/15/15 split
        if min_class_count >= 6:
            # Split into train/val/test (70/15/15)
            train_results, temp_results = train_test_split(
                all_results, test_size=0.3, random_state=42,
                stratify=labels
            )
            val_results, test_results = train_test_split(
                temp_results, test_size=0.5, random_state=42,
                stratify=[r['label'] for r in temp_results]
            )
        else:
            # For small datasets, use simple split without stratification
            print(f"  ⚠️  Small dataset ({len(all_results)} samples), using non-stratified split")
            train_results, temp_results = train_test_split(
                all_results, test_size=0.3, random_state=42
            )
            val_results, test_results = train_test_split(
                temp_results, test_size=0.5, random_state=42
            )
        
        # Select appropriate split
        if self.split == 'train':
            self.data = train_results
        elif self.split == 'val':
            self.data = val_results
        elif self.split == 'test':
            self.data = test_results
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns dictionary with:
            - rppg_signals: (num_regions, 3, sequence_length)
            - labels: scalar
            - handcrafted_features: (feature_dim,) [optional]
            - video_name: str
        """
        sample = self.data[idx]
        
        # Extract rPPG signals from pulse results
        # We need the original RGB signals, not the extracted pulse
        # For now, we'll use the pulse signals and stack them
        # In production, you'd save the original RGB signals
        
        # Get pulse results
        pulse_results = sample['pulse_results']
        
        # Initialize rPPG signals tensor
        rppg_signals = np.zeros((self.num_regions, 3, self.sequence_length))
        
        # This is a workaround - ideally we'd have the raw RGB signals saved
        # For now, we'll simulate by using the pulse signal
        for roi_idx, (roi_name, result) in enumerate(pulse_results.items()):
            if roi_idx >= self.num_regions:
                break
            # Use pulse signal for all 3 channels (not ideal but works for testing)
            pulse_signal = result.get('pulse_signal', np.zeros(self.sequence_length))
            if len(pulse_signal) < self.sequence_length:
                pulse_signal = np.pad(pulse_signal, (0, self.sequence_length - len(pulse_signal)))
            elif len(pulse_signal) > self.sequence_length:
                pulse_signal = pulse_signal[:self.sequence_length]
            
            # Normalize
            if np.std(pulse_signal) > 0:
                pulse_signal = (pulse_signal - np.mean(pulse_signal)) / np.std(pulse_signal)
            
            # Replicate for RGB channels (temporary solution)
            for c in range(3):
                rppg_signals[roi_idx, c, :] = pulse_signal
        
        # Convert to tensor
        rppg_signals = torch.FloatTensor(rppg_signals)
        
        # Augmentation (if enabled)
        if self.augment:
            rppg_signals = self._augment(rppg_signals)
        
        # Label
        label = torch.LongTensor([sample['label']])[0]
        
        # Output dictionary
        output = {
            'rppg_signals': rppg_signals,
            'labels': label,
            'video_name': sample['video_name']
        }
        
        # Hand-crafted features
        if self.use_handcrafted:
            features = sample['features']
            # Convert to array
            feature_values = np.array([v for v in features.values()], dtype=np.float32)
            # Replace inf/nan with large value
            feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=999.0, neginf=-999.0)
            output['handcrafted_features'] = torch.FloatTensor(feature_values)
        
        return output
    
    def _augment(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to rPPG signals
        
        Args:
            signal: (num_regions, 3, sequence_length)
        
        Returns:
            Augmented signal
        """
        # Random noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(signal) * 0.05
            signal = signal + noise
        
        # Random scaling
        if torch.rand(1) < 0.3:
            scale = torch.FloatTensor([1.0]).uniform_(0.9, 1.1)
            signal = signal * scale
        
        # Random temporal shift (circular)
        if torch.rand(1) < 0.3:
            shift = torch.randint(-10, 10, (1,)).item()
            signal = torch.roll(signal, shifts=shift, dims=-1)
        
        return signal


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_handcrafted: bool = True,
    sequence_length: int = 150,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_path: Path to processed features pickle file
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_handcrafted: Whether to use hand-crafted features
        sequence_length: Sequence length
        augment_train: Whether to augment training data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = RPPGDataset(
        data_path=data_path,
        split='train',
        use_handcrafted=use_handcrafted,
        sequence_length=sequence_length,
        augment=augment_train
    )
    
    val_dataset = RPPGDataset(
        data_path=data_path,
        split='val',
        use_handcrafted=use_handcrafted,
        sequence_length=sequence_length,
        augment=False
    )
    
    test_dataset = RPPGDataset(
        data_path=data_path,
        split='test',
        use_handcrafted=use_handcrafted,
        sequence_length=sequence_length,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATASET LOADER - TESTING")
    print("="*70 + "\n")
    
    # Test with the processed data
    data_path = "../../outputs/processed_features/dataset_features_chrom.pkl"
    
    if Path(data_path).exists():
        print("Testing dataset loader...")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            batch_size=2,
            num_workers=0,
            use_handcrafted=True
        )
        
        # Test iteration
        print("Testing batch loading...")
        for batch in train_loader:
            print(f"  rPPG signals: {batch['rppg_signals'].shape}")
            print(f"  Labels: {batch['labels'].shape}")
            print(f"  Handcrafted: {batch['handcrafted_features'].shape}")
            print(f"  Video names: {batch['video_name']}")
            break
        
        print("\n✅ Dataset loader working correctly!")
    else:
        print(f"⚠️  Data file not found: {data_path}")
        print("Run preprocessing first: python src/preprocessing/process_dataset.py")
    
    print("\n" + "="*70 + "\n")

