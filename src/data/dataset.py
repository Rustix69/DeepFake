"""
PyTorch Dataset classes for Celeb-DF v2 deepfake detection
Supports frame extraction, ROI extraction, and rPPG signal processing
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json


class CelebDFDataset(Dataset):
    """
    PyTorch Dataset for Celeb-DF v2 deepfake detection
    
    Args:
        dataset_root: Root directory of Celeb-DF-v2 dataset
        split: 'train', 'val', or 'test'
        frame_count: Number of frames to sample from each video
        transform: Optional transform to apply to frames
        return_video: If True, return all frames; if False, return single frame
    """
    
    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        frame_count: int = 30,
        transform=None,
        return_video: bool = True,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.frame_count = frame_count
        self.transform = transform
        self.return_video = return_video
        self.frame_size = frame_size
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
        # Load train/test split
        self.data = self._load_split()
        
        print(f"Loaded {len(self.data)} videos for {split} split")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata from CSV"""
        metadata_path = Path("outputs/dataset_analysis.csv")
        if metadata_path.exists():
            return pd.read_csv(metadata_path)
        else:
            raise FileNotFoundError("Dataset metadata not found. Run dataset_explorer.py first.")
    
    def _load_split(self) -> pd.DataFrame:
        """Load train/val/test split"""
        # Load test videos list
        test_list_path = self.dataset_root / "List_of_testing_videos.txt"
        test_videos = set()
        
        with open(test_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    video_path = parts[1]
                    test_videos.add(video_path)
        
        # Filter metadata based on split
        if self.split == 'test':
            data = self.metadata[self.metadata['is_test'] == True].copy()
        elif self.split == 'train':
            # Use 90% of non-test videos for training
            non_test = self.metadata[self.metadata['is_test'] == False].copy()
            train_size = int(len(non_test) * 0.9)
            data = non_test.iloc[:train_size]
        elif self.split == 'val':
            # Use 10% of non-test videos for validation
            non_test = self.metadata[self.metadata['is_test'] == False].copy()
            train_size = int(len(non_test) * 0.9)
            data = non_test.iloc[train_size:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        return data.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary containing:
                - frames: Tensor of shape (T, C, H, W) or (C, H, W)
                - label: 0 for fake, 1 for real
                - video_path: Path to video file
                - metadata: Additional metadata
        """
        row = self.data.iloc[idx]
        video_path = self.dataset_root / row['relative_path']
        label = int(row['label'])
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        if frames is None or len(frames) == 0:
            # Return zeros if video fails to load
            if self.return_video:
                frames = torch.zeros(self.frame_count, 3, *self.frame_size)
            else:
                frames = torch.zeros(3, *self.frame_size)
        else:
            # Apply transforms if provided
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            
            # Stack frames
            frames = torch.stack(frames) if self.return_video else frames[0]
        
        return {
            'frames': frames,
            'label': torch.tensor(label, dtype=torch.long),
            'video_path': str(video_path),
            'filename': row['filename'],
            'category': row['category'],
            'duration': row['duration'],
            'fps': row['fps']
        }
    
    def _extract_frames(self, video_path: Path) -> Optional[List[torch.Tensor]]:
        """Extract uniformly sampled frames from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return None
            
            # Calculate frame indices to sample
            if total_frames <= self.frame_count:
                # If video has fewer frames, sample all and repeat last
                frame_indices = list(range(total_frames))
                frame_indices += [total_frames - 1] * (self.frame_count - total_frames)
            else:
                # Uniformly sample frames
                frame_indices = np.linspace(0, total_frames - 1, self.frame_count, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame
                    frame = cv2.resize(frame, self.frame_size)
                    
                    # Convert to tensor and normalize
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    
                    frames.append(frame)
                else:
                    # If frame read fails, use last valid frame or zeros
                    if len(frames) > 0:
                        frames.append(frames[-1])
                    else:
                        frames.append(torch.zeros(3, *self.frame_size))
            
            cap.release()
            
            return frames if self.return_video else [frames[0]]
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None


class CelebDFFrameDataset(Dataset):
    """
    Dataset that returns individual frames instead of video sequences
    Useful for frame-level analysis and faster iteration
    """
    
    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        frames_per_video: int = 10,
        transform=None,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.frame_size = frame_size
        
        # Create frame-level index
        self.frame_index = self._create_frame_index()
        
        print(f"Loaded {len(self.frame_index)} frames for {split} split")
    
    def _create_frame_index(self) -> List[Dict]:
        """Create index of (video_path, frame_idx, label) tuples"""
        # First create video dataset to get split
        video_dataset = CelebDFDataset(
            self.dataset_root,
            split=self.split,
            return_video=False
        )
        
        frame_index = []
        for idx in range(len(video_dataset)):
            row = video_dataset.data.iloc[idx]
            video_path = self.dataset_root / row['relative_path']
            label = int(row['label'])
            
            # Create entries for each frame
            for frame_idx in range(self.frames_per_video):
                frame_index.append({
                    'video_path': video_path,
                    'frame_idx': frame_idx,
                    'label': label,
                    'category': row['category']
                })
        
        return frame_index
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict:
        entry = self.frame_index[idx]
        
        # Extract specific frame
        frame = self._extract_single_frame(
            entry['video_path'],
            entry['frame_idx']
        )
        
        if frame is None:
            frame = torch.zeros(3, *self.frame_size)
        elif self.transform:
            frame = self.transform(frame)
        
        return {
            'frame': frame,
            'label': torch.tensor(entry['label'], dtype=torch.long),
            'video_path': str(entry['video_path']),
            'frame_idx': entry['frame_idx'],
            'category': entry['category']
        }
    
    def _extract_single_frame(self, video_path: Path, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract a single frame from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return None
            
            # Calculate actual frame position
            actual_frame = int((frame_idx / self.frames_per_video) * total_frames)
            actual_frame = min(actual_frame, total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
            ret, frame = cap.read()
            
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                
                # Convert to tensor and normalize
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                
                return frame
            else:
                return None
                
        except Exception as e:
            print(f"Error loading frame from {video_path}: {e}")
            return None


def create_dataloaders(
    dataset_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    frame_count: int = 30,
    frame_size: Tuple[int, int] = (224, 224),
    use_frame_dataset: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        dataset_root: Root directory of dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        frame_count: Number of frames per video
        frame_size: Size to resize frames to
        use_frame_dataset: Use frame-level dataset instead of video dataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    DatasetClass = CelebDFFrameDataset if use_frame_dataset else CelebDFDataset
    
    # Create datasets
    train_dataset = DatasetClass(
        dataset_root=dataset_root,
        split='train',
        frame_size=frame_size,
        **({'frames_per_video': frame_count} if use_frame_dataset else {'frame_count': frame_count})
    )
    
    val_dataset = DatasetClass(
        dataset_root=dataset_root,
        split='val',
        frame_size=frame_size,
        **({'frames_per_video': frame_count} if use_frame_dataset else {'frame_count': frame_count})
    )
    
    test_dataset = DatasetClass(
        dataset_root=dataset_root,
        split='test',
        frame_size=frame_size,
        **({'frames_per_video': frame_count} if use_frame_dataset else {'frame_count': frame_count})
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
    
    return train_loader, val_loader, test_loader


def get_dataset_statistics(dataset_root: str) -> Dict:
    """Load dataset statistics from saved JSON"""
    stats_path = Path("outputs/dataset_statistics.json")
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError("Dataset statistics not found. Run dataset_explorer.py first.")


if __name__ == "__main__":
    # Test dataset loading
    print("Testing CelebDFDataset...")
    dataset_root = "/home/traderx/DeepFake/Celeb-DF-v2"
    
    # Test video dataset
    train_dataset = CelebDFDataset(
        dataset_root=dataset_root,
        split='train',
        frame_count=10,
        return_video=True
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    
    # Test loading first sample
    sample = train_dataset[0]
    print(f"Sample frames shape: {sample['frames'].shape}")
    print(f"Sample label: {sample['label']}")
    print(f"Sample video: {sample['filename']}")
    print(f"Sample category: {sample['category']}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=4,
        num_workers=2,
        frame_count=10
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch frames shape: {batch['frames'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    print(f"Batch labels: {batch['label']}")
    
    print("\n✅ Dataset loading test completed successfully!")

