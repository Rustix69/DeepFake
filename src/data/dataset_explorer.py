"""
Dataset Exploration and Validation Script
Analyzes Celeb-DF v2 dataset structure, video properties, and integrity
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class CelebDFExplorer:
    """Comprehensive dataset explorer for Celeb-DF v2"""
    
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.celeb_real = self.dataset_root / "Celeb-real"
        self.celeb_synth = self.dataset_root / "Celeb-synthesis"
        self.youtube_real = self.dataset_root / "YouTube-real"
        self.test_list = self.dataset_root / "List_of_testing_videos.txt"
        
        self.video_info = []
        self.stats = {}
        
    def load_test_split(self):
        """Load official test split"""
        test_videos = {}
        with open(self.test_list, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    label = int(parts[0])
                    video_path = parts[1]
                    test_videos[video_path] = label
        return test_videos
    
    def get_video_properties(self, video_path):
        """Extract properties from a video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file size
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            
            # Try to read first frame to validate
            ret, frame = cap.read()
            is_valid = ret and frame is not None
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size_mb': file_size,
                'resolution': f"{width}x{height}",
                'is_valid': is_valid
            }
        except Exception as e:
            return {
                'fps': 0,
                'frame_count': 0,
                'width': 0,
                'height': 0,
                'duration': 0,
                'file_size_mb': 0,
                'resolution': 'unknown',
                'is_valid': False,
                'error': str(e)
            }
    
    def explore_directory(self, directory, label, category):
        """Explore all videos in a directory"""
        video_files = list(directory.glob("*.mp4"))
        results = []
        
        print(f"\nAnalyzing {category} ({len(video_files)} videos)...")
        
        for video_path in tqdm(video_files, desc=f"Processing {category}"):
            props = self.get_video_properties(video_path)
            
            # Get relative path for test split matching
            rel_path = str(video_path.relative_to(self.dataset_root))
            
            info = {
                'filename': video_path.name,
                'relative_path': rel_path,
                'label': label,  # 1 for real, 0 for fake
                'category': category,
                **props
            }
            
            results.append(info)
            self.video_info.append(info)
        
        return results
    
    def analyze_dataset(self):
        """Perform comprehensive dataset analysis"""
        print("=" * 60)
        print("CELEB-DF V2 DATASET EXPLORATION")
        print("=" * 60)
        
        # Explore all directories
        celeb_real_results = self.explore_directory(self.celeb_real, 1, "Celeb-Real")
        youtube_real_results = self.explore_directory(self.youtube_real, 1, "YouTube-Real")
        celeb_synth_results = self.explore_directory(self.celeb_synth, 0, "Celeb-Synthesis")
        
        # Load test split
        test_videos = self.load_test_split()
        
        # Mark test videos
        for video in self.video_info:
            video['is_test'] = video['relative_path'] in test_videos
        
        # Create DataFrame
        df = pd.DataFrame(self.video_info)
        
        # Calculate statistics
        self.calculate_statistics(df)
        
        # Print summary
        self.print_summary(df)
        
        return df
    
    def calculate_statistics(self, df):
        """Calculate dataset statistics"""
        self.stats = {
            'total_videos': len(df),
            'real_videos': len(df[df['label'] == 1]),
            'fake_videos': len(df[df['label'] == 0]),
            'test_videos': len(df[df['is_test'] == True]),
            'train_videos': len(df[df['is_test'] == False]),
            'valid_videos': len(df[df['is_valid'] == True]),
            'invalid_videos': len(df[df['is_valid'] == False]),
            
            # Video properties statistics
            'avg_duration': df['duration'].mean(),
            'avg_fps': df['fps'].mean(),
            'avg_frame_count': df['frame_count'].mean(),
            'avg_file_size_mb': df['file_size_mb'].mean(),
            
            # Resolution distribution
            'resolutions': df['resolution'].value_counts().to_dict(),
            
            # Category breakdown
            'category_counts': df['category'].value_counts().to_dict(),
        }
    
    def print_summary(self, df):
        """Print comprehensive dataset summary"""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 Video Count:")
        print(f"   Total Videos:        {self.stats['total_videos']:,}")
        print(f"   Real Videos:         {self.stats['real_videos']:,}")
        print(f"   Fake Videos:         {self.stats['fake_videos']:,}")
        print(f"   Real:Fake Ratio:     1:{self.stats['fake_videos']/self.stats['real_videos']:.2f}")
        
        print(f"\n🔍 Data Split:")
        print(f"   Training Videos:     {self.stats['train_videos']:,}")
        print(f"   Test Videos:         {self.stats['test_videos']:,}")
        
        print(f"\n✅ Data Quality:")
        print(f"   Valid Videos:        {self.stats['valid_videos']:,}")
        print(f"   Invalid Videos:      {self.stats['invalid_videos']:,}")
        
        print(f"\n📹 Video Properties (Average):")
        print(f"   Duration:            {self.stats['avg_duration']:.2f} seconds")
        print(f"   FPS:                 {self.stats['avg_fps']:.2f}")
        print(f"   Frame Count:         {self.stats['avg_frame_count']:.0f}")
        print(f"   File Size:           {self.stats['avg_file_size_mb']:.2f} MB")
        
        print(f"\n📐 Resolution Distribution:")
        for resolution, count in sorted(self.stats['resolutions'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {resolution:15} {count:5,} videos ({count/self.stats['total_videos']*100:.1f}%)")
        
        print(f"\n📁 Category Breakdown:")
        for category, count in self.stats['category_counts'].items():
            print(f"   {category:20} {count:5,} videos")
        
        print("\n" + "=" * 60)
    
    def save_results(self, df, output_dir="outputs"):
        """Save analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / "dataset_analysis.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved dataset analysis to: {csv_path}")
        
        # Save JSON statistics
        json_path = output_dir / "dataset_statistics.json"
        with open(json_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"💾 Saved statistics to: {json_path}")
        
        return csv_path, json_path
    
    def plot_visualizations(self, df, output_dir="outputs/figures"):
        """Create visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Label Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Real vs Fake
        label_counts = df['label'].value_counts()
        axes[0, 0].bar(['Real', 'Fake'], [label_counts.get(1, 0), label_counts.get(0, 0)], 
                      color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('Real vs Fake Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Duration distribution
        df[df['is_valid']]['duration'].hist(bins=50, ax=axes[0, 1], alpha=0.7, color='blue')
        axes[0, 1].set_title('Video Duration Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('Count')
        
        # FPS distribution
        df[df['is_valid']]['fps'].hist(bins=30, ax=axes[1, 0], alpha=0.7, color='orange')
        axes[1, 0].set_title('FPS Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('FPS')
        axes[1, 0].set_ylabel('Count')
        
        # File size distribution
        df[df['is_valid']]['file_size_mb'].hist(bins=50, ax=axes[1, 1], alpha=0.7, color='purple')
        axes[1, 1].set_title('File Size Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('File Size (MB)')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plot_path = output_dir / "dataset_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved visualization to: {plot_path}")
        plt.close()
        
        # 2. Category comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Duration by category
        df[df['is_valid']].boxplot(column='duration', by='category', ax=axes[0])
        axes[0].set_title('Duration by Category', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Duration (seconds)')
        plt.sca(axes[0])
        plt.xticks(rotation=45, ha='right')
        
        # File size by category
        df[df['is_valid']].boxplot(column='file_size_mb', by='category', ax=axes[1])
        axes[1].set_title('File Size by Category', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('File Size (MB)')
        plt.sca(axes[1])
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = output_dir / "category_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved visualization to: {plot_path}")
        plt.close()


def main():
    """Main execution function"""
    dataset_root = "/home/traderx/DeepFake/Celeb-DF-v2"
    
    # Create explorer
    explorer = CelebDFExplorer(dataset_root)
    
    # Analyze dataset
    df = explorer.analyze_dataset()
    
    # Save results
    explorer.save_results(df)
    
    # Create visualizations
    explorer.plot_visualizations(df)
    
    print("\n✅ Dataset exploration completed successfully!")
    print(f"📁 Results saved to: outputs/")
    
    return df, explorer


if __name__ == "__main__":
    df, explorer = main()

