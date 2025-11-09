"""
Data Split Management for Celeb-DF v2
Manages train/val/test splits and provides statistics
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


class DataSplitManager:
    """Manage and analyze data splits for Celeb-DF v2 dataset"""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.metadata = self._load_metadata()
        self.test_videos = self._load_test_list()
        self.splits = self._create_splits()
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata"""
        metadata_path = Path("outputs/dataset_analysis.csv")
        return pd.read_csv(metadata_path)
    
    def _load_test_list(self) -> set:
        """Load official test split videos"""
        test_list_path = self.dataset_root / "List_of_testing_videos.txt"
        test_videos = set()
        
        with open(test_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    test_videos.add(parts[1])
        
        return test_videos
    
    def _create_splits(self) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits"""
        # Separate test set
        test_mask = self.metadata['is_test'] == True
        test_df = self.metadata[test_mask].copy()
        
        # Split remaining into train/val (90/10)
        non_test = self.metadata[~test_mask].copy()
        train_size = int(len(non_test) * 0.9)
        
        train_df = non_test.iloc[:train_size].copy()
        val_df = non_test.iloc[train_size:].copy()
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def get_split_statistics(self) -> Dict:
        """Get comprehensive statistics for each split"""
        stats = {}
        
        for split_name, split_df in self.splits.items():
            real_count = len(split_df[split_df['label'] == 1])
            fake_count = len(split_df[split_df['label'] == 0])
            
            stats[split_name] = {
                'total_videos': len(split_df),
                'real_videos': real_count,
                'fake_videos': fake_count,
                'real_ratio': real_count / len(split_df) if len(split_df) > 0 else 0,
                'fake_ratio': fake_count / len(split_df) if len(split_df) > 0 else 0,
                'category_distribution': split_df['category'].value_counts().to_dict(),
                'avg_duration': split_df['duration'].mean(),
                'avg_fps': split_df['fps'].mean(),
                'avg_frame_count': split_df['frame_count'].mean(),
            }
        
        return stats
    
    def print_summary(self):
        """Print formatted split summary"""
        print("=" * 70)
        print("DATA SPLIT SUMMARY")
        print("=" * 70)
        
        stats = self.get_split_statistics()
        
        for split_name in ['train', 'val', 'test']:
            split_stats = stats[split_name]
            
            print(f"\n{'='*70}")
            print(f"{split_name.upper()} SPLIT")
            print(f"{'='*70}")
            
            print(f"\n📊 Video Count:")
            print(f"   Total:          {split_stats['total_videos']:>6,}")
            print(f"   Real:           {split_stats['real_videos']:>6,}  ({split_stats['real_ratio']*100:5.1f}%)")
            print(f"   Fake:           {split_stats['fake_videos']:>6,}  ({split_stats['fake_ratio']*100:5.1f}%)")
            
            print(f"\n📁 Category Distribution:")
            for category, count in sorted(split_stats['category_distribution'].items()):
                percentage = (count / split_stats['total_videos']) * 100
                print(f"   {category:25} {count:>5,}  ({percentage:5.1f}%)")
            
            print(f"\n📹 Video Properties:")
            print(f"   Avg Duration:   {split_stats['avg_duration']:>6.2f} seconds")
            print(f"   Avg FPS:        {split_stats['avg_fps']:>6.2f}")
            print(f"   Avg Frames:     {split_stats['avg_frame_count']:>6.0f}")
        
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")
        
        total_train = stats['train']['total_videos']
        total_val = stats['val']['total_videos']
        total_test = stats['test']['total_videos']
        total_all = total_train + total_val + total_test
        
        print(f"\nTotal Videos:    {total_all:>6,}")
        print(f"Train:           {total_train:>6,}  ({total_train/total_all*100:5.1f}%)")
        print(f"Validation:      {total_val:>6,}  ({total_val/total_all*100:5.1f}%)")
        print(f"Test:            {total_test:>6,}  ({total_test/total_all*100:5.1f}%)")
        
        print(f"\n{'='*70}")
    
    def save_splits(self, output_dir: str = "outputs"):
        """Save split information to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save split statistics
        stats = self.get_split_statistics()
        stats_path = output_dir / "split_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Saved split statistics to: {stats_path}")
        
        # Save each split to CSV
        for split_name, split_df in self.splits.items():
            csv_path = output_dir / f"{split_name}_split.csv"
            split_df.to_csv(csv_path, index=False)
            print(f"💾 Saved {split_name} split to: {csv_path}")
    
    def get_split(self, split_name: str) -> pd.DataFrame:
        """Get specific split DataFrame"""
        return self.splits[split_name]
    
    def verify_no_overlap(self) -> bool:
        """Verify that splits don't overlap"""
        train_files = set(self.splits['train']['relative_path'])
        val_files = set(self.splits['val']['relative_path'])
        test_files = set(self.splits['test']['relative_path'])
        
        train_val_overlap = train_files & val_files
        train_test_overlap = train_files & test_files
        val_test_overlap = val_files & test_files
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("⚠️  WARNING: Overlapping videos detected!")
            if train_val_overlap:
                print(f"   Train-Val overlap: {len(train_val_overlap)} videos")
            if train_test_overlap:
                print(f"   Train-Test overlap: {len(train_test_overlap)} videos")
            if val_test_overlap:
                print(f"   Val-Test overlap: {len(val_test_overlap)} videos")
            return False
        else:
            print("✅ No overlap detected between splits")
            return True


def main():
    """Test split manager"""
    dataset_root = "/home/traderx/DeepFake/Celeb-DF-v2"
    
    manager = DataSplitManager(dataset_root)
    
    # Print summary
    manager.print_summary()
    
    # Verify no overlap
    print("\n")
    manager.verify_no_overlap()
    
    # Save splits
    manager.save_splits()


if __name__ == "__main__":
    main()

