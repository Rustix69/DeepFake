"""
Simple script to visualize face detection and ROI extraction results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from face_detection import FaceDetector
from roi_extraction import ROIExtractor


def visualize_single_video(video_path: str, frame_index: int = 0):
    """
    Visualize face detection and ROI extraction on a single frame
    
    Args:
        video_path: Path to video file
        frame_index: Frame index to visualize
    """
    # Initialize
    face_detector = FaceDetector()
    roi_extractor = ROIExtractor()
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_index} from {video_path}")
        return
    
    # Detect face
    landmarks = face_detector.detect_face(frame)
    
    if not landmarks:
        print(f"❌ No face detected in frame {frame_index}")
        return
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # 1. Original frame
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Frame', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Frame with landmarks
    landmark_frame = face_detector.draw_landmarks(frame, landmarks)
    axes[1].imshow(cv2.cvtColor(landmark_frame, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Face with 478 Landmarks', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Frame with ROIs
    roi_coords = roi_extractor.extract_roi_coordinates(landmarks, frame.shape)
    roi_frame = roi_extractor.draw_rois(frame, roi_coords, show_labels=True)
    axes[2].imshow(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Extracted ROIs (7 regions)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Print ROI statistics
    print(f"\n📊 ROI Mean RGB Values:")
    print("=" * 60)
    for roi_name, coords in roi_coords.items():
        mean_color = roi_extractor.get_roi_mean_color(frame, coords)
        print(f"{roi_name:15} | R: {mean_color[0]:6.2f} | G: {mean_color[1]:6.2f} | B: {mean_color[2]:6.2f}")
    
    # Save figure
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    output_file = output_dir / f"{video_name}_frame_{frame_index:04d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization to: {output_file}")
    
    plt.show()


def compare_real_vs_fake():
    """Compare real and fake video side by side"""
    # Initialize
    face_detector = FaceDetector()
    roi_extractor = ROIExtractor()
    
    # Paths
    real_video = "../../Celeb-DF-v2/Celeb-real/id0_0000.mp4"
    fake_video = "../../Celeb-DF-v2/Celeb-synthesis/id0_id1_0000.mp4"
    
    # Process real video
    cap_real = cv2.VideoCapture(real_video)
    ret_real, frame_real = cap_real.read()
    cap_real.release()
    
    # Process fake video (skip to middle where face is likely)
    cap_fake = cv2.VideoCapture(fake_video)
    total_frames = int(cap_fake.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_fake.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret_fake, frame_fake = cap_fake.read()
    cap_fake.release()
    
    if not (ret_real and ret_fake):
        print("❌ Failed to load videos")
        return
    
    # Detect and visualize
    landmarks_real = face_detector.detect_face(frame_real)
    landmarks_fake = face_detector.detect_face(frame_fake)
    
    if not (landmarks_real and landmarks_fake):
        print("❌ Face detection failed")
        return
    
    # Extract ROIs
    roi_coords_real = roi_extractor.extract_roi_coordinates(landmarks_real, frame_real.shape)
    roi_coords_fake = roi_extractor.extract_roi_coordinates(landmarks_fake, frame_fake.shape)
    
    roi_frame_real = roi_extractor.draw_rois(frame_real, roi_coords_real, show_labels=True)
    roi_frame_fake = roi_extractor.draw_rois(frame_fake, roi_coords_fake, show_labels=True)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(cv2.cvtColor(roi_frame_real, cv2.COLOR_BGR2RGB))
    axes[0].set_title('REAL VIDEO - ROI Extraction', fontsize=16, fontweight='bold', color='green')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(roi_frame_fake, cv2.COLOR_BGR2RGB))
    axes[1].set_title('FAKE VIDEO - ROI Extraction', fontsize=16, fontweight='bold', color='red')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "real_vs_fake_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved comparison to: {output_file}")
    
    plt.show()
    
    # Print comparison statistics
    print(f"\n📊 Comparison - Mean RGB Values:")
    print("=" * 80)
    print(f"{'ROI':15} | {'Real (R/G/B)':30} | {'Fake (R/G/B)':30} | {'Difference':15}")
    print("=" * 80)
    
    for roi_name in roi_coords_real.keys():
        real_color = roi_extractor.get_roi_mean_color(frame_real, roi_coords_real[roi_name])
        fake_color = roi_extractor.get_roi_mean_color(frame_fake, roi_coords_fake[roi_name])
        diff = np.abs(real_color - fake_color).mean()
        
        print(f"{roi_name:15} | {real_color[0]:5.1f}/{real_color[1]:5.1f}/{real_color[2]:5.1f} | "
              f"{fake_color[0]:5.1f}/{fake_color[1]:5.1f}/{fake_color[2]:5.1f} | {diff:8.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VISUALIZATION SCRIPT")
    print("=" * 70)
    
    print("\n1️⃣  Visualizing Real Video...")
    visualize_single_video("../../Celeb-DF-v2/Celeb-real/id0_0000.mp4", frame_index=0)
    
    print("\n2️⃣  Visualizing Fake Video...")
    visualize_single_video("../../Celeb-DF-v2/Celeb-synthesis/id0_id1_0000.mp4", frame_index=280)
    
    print("\n3️⃣  Comparing Real vs Fake...")
    compare_real_vs_fake()
    
    print("\n" + "=" * 70)
    print("✅ All visualizations complete!")
    print("📁 Check outputs/visualizations/ for saved images")
    print("=" * 70)

