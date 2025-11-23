"""
Dataset Processing Script
Process entire Celeb-DF-v2 dataset and extract rPPG features
Saves features for model training
"""

import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import json
import cv2

try:
    from .face_detection import FaceDetector
    from .roi_extraction import ROIExtractor
    from .rppg_extraction import RPPGExtractor
    from .feature_extraction import DeepfakeFeatureExtractor, compute_deepfake_score
except ImportError:
    from face_detection import FaceDetector
    from roi_extraction import ROIExtractor
    from rppg_extraction import RPPGExtractor
    from feature_extraction import DeepfakeFeatureExtractor, compute_deepfake_score


class DatasetProcessor:
    """
    Process entire video dataset and extract rPPG features
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "../../outputs/processed_features",
        num_frames: int = 150,
        method: str = 'chrom'
    ):
        """
        Initialize dataset processor
        
        Args:
            dataset_path: Path to Celeb-DF-v2 dataset
            output_dir: Directory to save processed features
            num_frames: Number of frames to process per video
            method: rPPG extraction method
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_frames = num_frames
        self.method = method
        
        # Initialize processors
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor(padding=5)
        
        print(f"✅ Initialized Dataset Processor")
        print(f"   Dataset: {dataset_path}")
        print(f"   Output: {output_dir}")
        print(f"   Frames per video: {num_frames}")
        print(f"   rPPG method: {method.upper()}")
    
    def process_single_video(
        self,
        video_path: str,
        label: int
    ) -> Dict:
        """
        Process single video and extract features
        
        Args:
            video_path: Path to video file
            label: 0 for real, 1 for fake
        
        Returns:
            Dictionary with features and metadata
        """
        try:
            # Get video info
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30.0
            cap.release()
            
            # Detect faces
            landmarks_list, frame_indices = self.face_detector.process_video_smart(
                video_path,
                sample_strategy='smart',
                max_frames=self.num_frames
            )
            
            faces_detected = sum(1 for l in landmarks_list if l is not None)
            
            # Skip if too few faces detected
            if faces_detected < 30:
                return None
            
            # Extract ROI signals
            roi_signals = {roi_name: [] for roi_name in self.roi_extractor.ROI_LANDMARKS.keys()}
            
            cap = cv2.VideoCapture(video_path)
            for frame_idx, landmarks in zip(frame_indices, landmarks_list):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or landmarks is None:
                    for roi_name in roi_signals.keys():
                        roi_signals[roi_name].append(np.array([0, 0, 0]))
                    continue
                
                roi_coords_dict = self.roi_extractor.extract_roi_coordinates(landmarks, frame.shape)
                
                for roi_name, roi_coords in roi_coords_dict.items():
                    mean_color = self.roi_extractor.get_roi_mean_color(frame, roi_coords)
                    roi_signals[roi_name].append(mean_color)
            
            cap.release()
            
            # Convert to arrays
            for roi_name in roi_signals.keys():
                roi_signals[roi_name] = np.array(roi_signals[roi_name])
            
            # Extract rPPG signals
            rppg_extractor = RPPGExtractor(fps=fps, method=self.method)
            pulse_results = rppg_extractor.extract_from_roi_signals(roi_signals)
            
            # Extract features
            feature_extractor = DeepfakeFeatureExtractor(fps=fps)
            features = feature_extractor.extract_all_features(pulse_results, include_per_roi=False)
            
            # Compute baseline score
            score, prediction = compute_deepfake_score(features)
            
            # Compile result
            result = {
                'video_path': str(video_path),
                'video_name': Path(video_path).name,
                'label': label,
                'label_name': 'real' if label == 0 else 'fake',
                'fps': float(fps),
                'faces_detected': faces_detected,
                'detection_rate': faces_detected / self.num_frames,
                'features': features,
                'baseline_score': float(score),
                'baseline_prediction': prediction,
                'pulse_results': {
                    roi_name: {
                        'heart_rate_bpm': result['heart_rate_bpm'],
                        'snr': result['quality']['snr'],
                        'quality_score': result['quality']['quality_score'],
                        'is_good_quality': result['quality']['is_good_quality']
                    }
                    for roi_name, result in pulse_results.items()
                }
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error processing {Path(video_path).name}: {e}")
            return None
    
    def process_dataset(
        self,
        max_videos_per_class: Optional[int] = None,
        save_format: str = 'pickle'
    ) -> pd.DataFrame:
        """
        Process entire dataset
        
        Args:
            max_videos_per_class: Limit number of videos per class (for testing)
            save_format: 'pickle' or 'csv'
        
        Returns:
            DataFrame with all results
        """
        print("\n" + "="*70)
        print("PROCESSING DATASET")
        print("="*70 + "\n")
        
        # Get video lists
        real_videos = sorted(list((self.dataset_path / "Celeb-real").glob("*.mp4")))
        fake_videos = sorted(list((self.dataset_path / "Celeb-synthesis").glob("*.mp4")))
        
        if max_videos_per_class:
            real_videos = real_videos[:max_videos_per_class]
            fake_videos = fake_videos[:max_videos_per_class]
        
        print(f"📹 Found {len(real_videos)} real videos")
        print(f"📹 Found {len(fake_videos)} fake videos")
        print(f"📹 Total: {len(real_videos) + len(fake_videos)} videos\n")
        
        # Process real videos
        print("Processing REAL videos...")
        real_results = []
        for video_path in tqdm(real_videos, desc="Real"):
            result = self.process_single_video(str(video_path), label=0)
            if result is not None:
                real_results.append(result)
        
        print(f"✅ Processed {len(real_results)}/{len(real_videos)} real videos\n")
        
        # Process fake videos
        print("Processing FAKE videos...")
        fake_results = []
        for video_path in tqdm(fake_videos, desc="Fake"):
            result = self.process_single_video(str(video_path), label=1)
            if result is not None:
                fake_results.append(result)
        
        print(f"✅ Processed {len(fake_results)}/{len(fake_videos)} fake videos\n")
        
        # Combine results
        all_results = real_results + fake_results
        
        # Save results
        print(f"💾 Saving results to {self.output_dir}...")
        
        # Save complete results (with pulse signals) as pickle
        with open(self.output_dir / f"dataset_features_{self.method}.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        print(f"   ✅ Saved: dataset_features_{self.method}.pkl")
        
        # Create DataFrame for analysis (without raw signals)
        df_data = []
        for result in all_results:
            row = {
                'video_name': result['video_name'],
                'label': result['label'],
                'label_name': result['label_name'],
                'fps': result['fps'],
                'detection_rate': result['detection_rate'],
                'baseline_score': result['baseline_score'],
                'baseline_prediction': result['baseline_prediction']
            }
            # Add features
            row.update(result['features'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        df.to_csv(self.output_dir / f"dataset_features_{self.method}.csv", index=False)
        print(f"   ✅ Saved: dataset_features_{self.method}.csv")
        
        # Save summary
        summary = self.generate_summary(df)
        with open(self.output_dir / f"dataset_summary_{self.method}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✅ Saved: dataset_summary_{self.method}.json")
        
        return df
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        real_df = df[df['label'] == 0]
        fake_df = df[df['label'] == 1]
        
        summary = {
            'total_videos': len(df),
            'real_videos': len(real_df),
            'fake_videos': len(fake_df),
            'method': self.method,
            'num_frames_per_video': self.num_frames,
            'baseline_accuracy': float(np.mean(
                (df['baseline_prediction'] == 'REAL') == (df['label'] == 0)
            )),
            'real_stats': {
                'hr_std_mean': float(real_df['roi_hr_std'].mean()),
                'hr_std_std': float(real_df['roi_hr_std'].std()),
                'num_good_rois_mean': float(real_df['roi_num_good_rois'].mean()),
                'avg_quality_mean': float(real_df['roi_avg_quality_score'].mean())
            },
            'fake_stats': {
                'hr_std_mean': float(fake_df['roi_hr_std'].mean()),
                'hr_std_std': float(fake_df['roi_hr_std'].std()),
                'num_good_rois_mean': float(fake_df['roi_num_good_rois'].mean()),
                'avg_quality_mean': float(fake_df['roi_avg_quality_score'].mean())
            }
        }
        
        return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Celeb-DF-v2 dataset")
    parser.add_argument('--dataset', type=str, 
                       default="/home/traderx/DeepFake/Celeb-DF-v2",
                       help='Path to dataset')
    parser.add_argument('--output', type=str,
                       default="../../outputs/processed_features",
                       help='Output directory')
    parser.add_argument('--num-videos', type=int, default=None,
                       help='Max videos per class (for testing)')
    parser.add_argument('--num-frames', type=int, default=150,
                       help='Frames per video')
    parser.add_argument('--method', type=str, default='chrom',
                       choices=['chrom', 'pos', 'ica'],
                       help='rPPG extraction method')
    
    args = parser.parse_args()
    
    # Process dataset
    processor = DatasetProcessor(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_frames=args.num_frames,
        method=args.method
    )
    
    df = processor.process_dataset(max_videos_per_class=args.num_videos)
    
    print("\n" + "="*70)
    print("✅ DATASET PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nProcessed: {len(df)} videos")
    print(f"Features extracted: {len([c for c in df.columns if c not in ['video_name', 'label', 'label_name']])} features")
    print(f"\nFiles saved to: {args.output}")
    print("="*70 + "\n")

