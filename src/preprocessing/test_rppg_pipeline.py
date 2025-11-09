"""
Complete rPPG Pipeline Testing
Tests the full pipeline: Face Detection → ROI Extraction → rPPG Signal → Analysis
Validates on real vs fake videos for deepfake detection
"""

import sys
sys.path.append('..')

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from face_detection import FaceDetector
from roi_extraction import ROIExtractor
from rppg_extraction import RPPGExtractor, compare_rppg_methods


def process_video_full_pipeline(
    video_path: str,
    num_frames: int = 150,  # 5 seconds at 30 FPS
    method: str = 'chrom'
) -> Dict:
    """
    Process single video through complete pipeline
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to process
        method: rPPG method ('chrom', 'pos', or 'ica')
    
    Returns:
        Dictionary with complete analysis results
    """
    print(f"\n{'='*70}")
    print(f"Processing: {Path(video_path).name}")
    print(f"{'='*70}")
    
    # Initialize detectors
    face_detector = FaceDetector()
    roi_extractor = ROIExtractor(padding=5)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        fps = 30.0  # Default
    
    print(f"📹 Video: {fps:.1f} FPS, {total_frames} frames")
    
    # Detect faces and extract ROI signals
    print(f"🔍 Detecting faces and extracting ROI signals...")
    
    landmarks_list, frame_indices = face_detector.process_video_smart(
        video_path,
        sample_strategy='smart',
        max_frames=num_frames
    )
    
    faces_detected = sum(1 for l in landmarks_list if l is not None)
    print(f"   Detected faces in {faces_detected}/{num_frames} frames ({faces_detected/num_frames*100:.1f}%)")
    
    if faces_detected < 30:
        print("❌ Insufficient face detections, skipping video")
        return None
    
    # Extract ROI RGB signals
    roi_signals = {roi_name: [] for roi_name in roi_extractor.ROI_LANDMARKS.keys()}
    
    cap = cv2.VideoCapture(video_path)
    for frame_idx, landmarks in zip(frame_indices, landmarks_list):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or landmarks is None:
            # Append zeros for missing data
            for roi_name in roi_signals.keys():
                roi_signals[roi_name].append(np.array([0, 0, 0]))
            continue
        
        # Extract ROI coordinates and mean colors
        roi_coords_dict = roi_extractor.extract_roi_coordinates(landmarks, frame.shape)
        
        for roi_name, roi_coords in roi_coords_dict.items():
            mean_color = roi_extractor.get_roi_mean_color(frame, roi_coords)
            roi_signals[roi_name].append(mean_color)
    
    cap.release()
    
    # Convert to arrays
    for roi_name in roi_signals.keys():
        roi_signals[roi_name] = np.array(roi_signals[roi_name])
    
    print(f"✅ Extracted RGB signals from {len(roi_signals)} ROIs")
    
    # Extract rPPG signals
    print(f"💓 Extracting rPPG signals using {method.upper()} method...")
    
    rppg_extractor = RPPGExtractor(fps=fps, method=method)
    pulse_results = rppg_extractor.extract_from_roi_signals(roi_signals)
    
    # Analyze cross-region consistency
    consistency = rppg_extractor.analyze_cross_region_consistency(pulse_results)
    
    print(f"\n📊 Results:")
    print(f"   Good quality ROIs: {consistency['num_good_quality_rois']}/7")
    print(f"   Heart rate std: {consistency['hr_std']:.2f} BPM")
    print(f"   Consistency score: {consistency['consistency_score']:.3f}")
    print(f"   Is consistent: {'✅ YES' if consistency['is_consistent'] else '❌ NO'}")
    
    if consistency['num_good_quality_rois'] > 0:
        print(f"\n💓 Heart rates by ROI:")
        for roi_name in consistency['good_quality_rois']:
            hr = pulse_results[roi_name]['heart_rate_bpm']
            snr = pulse_results[roi_name]['quality']['snr']
            print(f"      {roi_name:15} {hr:6.1f} BPM (SNR: {snr:5.1f} dB)")
    
    return {
        'video_path': video_path,
        'fps': fps,
        'faces_detected': faces_detected,
        'roi_signals': roi_signals,
        'pulse_results': pulse_results,
        'consistency': consistency,
        'method': method
    }


def compare_real_vs_fake(
    real_videos: List[str],
    fake_videos: List[str],
    num_frames: int = 150,
    method: str = 'chrom'
) -> Dict:
    """
    Compare rPPG consistency between real and fake videos
    
    Args:
        real_videos: List of real video paths
        fake_videos: List of fake video paths
        num_frames: Frames to process per video
        method: rPPG method to use
    
    Returns:
        Comparison results
    """
    print("\n" + "="*70)
    print("REAL VS FAKE VIDEO COMPARISON")
    print("="*70)
    
    # Process real videos
    print("\n📹 Processing REAL videos...")
    real_results = []
    for video_path in real_videos:
        result = process_video_full_pipeline(video_path, num_frames, method)
        if result is not None:
            real_results.append(result)
    
    # Process fake videos
    print("\n📹 Processing FAKE videos...")
    fake_results = []
    for video_path in fake_videos:
        result = process_video_full_pipeline(video_path, num_frames, method)
        if result is not None:
            fake_results.append(result)
    
    # Analyze results
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    real_consistency_scores = [r['consistency']['consistency_score'] for r in real_results]
    fake_consistency_scores = [r['consistency']['consistency_score'] for r in fake_results]
    
    real_hr_stds = [r['consistency']['hr_std'] for r in real_results]
    fake_hr_stds = [r['consistency']['hr_std'] for r in fake_results]
    
    real_good_rois = [r['consistency']['num_good_quality_rois'] for r in real_results]
    fake_good_rois = [r['consistency']['num_good_quality_rois'] for r in fake_results]
    
    print(f"\n📊 Real Videos (n={len(real_results)}):")
    print(f"   Consistency Score: {np.mean(real_consistency_scores):.3f} ± {np.std(real_consistency_scores):.3f}")
    print(f"   HR Std Dev:        {np.mean(real_hr_stds):.2f} ± {np.std(real_hr_stds):.2f} BPM")
    print(f"   Good Quality ROIs: {np.mean(real_good_rois):.1f} ± {np.std(real_good_rois):.1f}")
    
    print(f"\n📊 Fake Videos (n={len(fake_results)}):")
    print(f"   Consistency Score: {np.mean(fake_consistency_scores):.3f} ± {np.std(fake_consistency_scores):.3f}")
    print(f"   HR Std Dev:        {np.mean(fake_hr_stds):.2f} ± {np.std(fake_hr_stds):.2f} BPM")
    print(f"   Good Quality ROIs: {np.mean(fake_good_rois):.1f} ± {np.std(fake_good_rois):.1f}")
    
    # Statistical significance
    print(f"\n📈 Differences:")
    diff_consistency = np.mean(real_consistency_scores) - np.mean(fake_consistency_scores)
    diff_hr_std = np.mean(fake_hr_stds) - np.mean(real_hr_stds)
    
    print(f"   Consistency Score: {diff_consistency:+.3f} (Real - Fake)")
    print(f"   HR Std Dev:        {diff_hr_std:+.2f} BPM (Fake - Real)")
    
    if diff_consistency > 0.1:
        print("   ✅ Real videos show HIGHER consistency (Good for detection!)")
    else:
        print("   ⚠️  No clear difference in consistency")
    
    if diff_hr_std > 5.0:
        print("   ✅ Fake videos show HIGHER variability (Good for detection!)")
    else:
        print("   ⚠️  No clear difference in variability")
    
    return {
        'real_results': real_results,
        'fake_results': fake_results,
        'real_consistency_scores': real_consistency_scores,
        'fake_consistency_scores': fake_consistency_scores,
        'real_hr_stds': real_hr_stds,
        'fake_hr_stds': fake_hr_stds
    }


def visualize_pulse_signals(
    result: Dict,
    output_path: str = "outputs/rppg_visualization"
):
    """
    Visualize extracted pulse signals
    
    Args:
        result: Result from process_video_full_pipeline
        output_path: Directory to save visualizations
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    pulse_results = result['pulse_results']
    fps = result['fps']
    video_name = Path(result['video_path']).stem
    
    # Create figure with subplots for each ROI
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    roi_names = list(pulse_results.keys())
    
    for idx, roi_name in enumerate(roi_names):
        if idx >= len(axes):
            break
        
        pulse_signal = pulse_results[roi_name]['pulse_signal']
        hr = pulse_results[roi_name]['heart_rate_bpm']
        snr = pulse_results[roi_name]['quality']['snr']
        is_good = pulse_results[roi_name]['quality']['is_good_quality']
        
        # Time axis
        time = np.arange(len(pulse_signal)) / fps
        
        # Plot
        color = 'green' if is_good else 'red'
        axes[idx].plot(time, pulse_signal, color=color, linewidth=1)
        axes[idx].set_title(f'{roi_name}\nHR: {hr:.1f} BPM | SNR: {snr:.1f} dB', 
                           fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplot
    if len(roi_names) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'rPPG Signals: {video_name}', fontsize=14, fontweight='bold', y=1.001)
    
    output_file = Path(output_path) / f"{video_name}_pulse_signals.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved visualization: {output_file}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE rPPG PIPELINE TEST")
    print("="*70)
    
    base_path = Path("/home/traderx/DeepFake/Celeb-DF-v2")
    
    # Get sample videos
    real_videos = sorted(list((base_path / "Celeb-real").glob("*.mp4")))[:3]
    fake_videos = sorted(list((base_path / "Celeb-synthesis").glob("*.mp4")))[:3]
    
    print(f"\nTesting with:")
    print(f"  • {len(real_videos)} real videos")
    print(f"  • {len(fake_videos)} fake videos")
    print(f"  • 150 frames per video (~5 seconds)")
    print(f"  • CHROM method (best for motion robustness)")
    
    # Run comparison
    comparison_results = compare_real_vs_fake(
        [str(v) for v in real_videos],
        [str(v) for v in fake_videos],
        num_frames=150,
        method='chrom'
    )
    
    # Visualize first real and fake video
    if comparison_results['real_results']:
        print("\n📊 Creating visualizations...")
        visualize_pulse_signals(comparison_results['real_results'][0])
    
    if comparison_results['fake_results']:
        visualize_pulse_signals(comparison_results['fake_results'][0])
    
    print("\n" + "="*70)
    print("✅ rPPG PIPELINE TEST COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use these rPPG features for model training")
    print("  2. Consistency score can be used as a detection feature")
    print("  3. Cross-region HR variance distinguishes real from fake")
    print("="*70 + "\n")

