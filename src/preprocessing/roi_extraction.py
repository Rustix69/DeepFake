"""
ROI (Region of Interest) Extraction from Facial Landmarks
Extracts physiologically significant facial regions for rPPG analysis
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ROIExtractor:
    """
    Extract Regions of Interest (ROIs) from facial landmarks
    
    Defines anatomically significant regions for rPPG signal extraction:
    - Forehead
    - Left cheek
    - Right cheek
    - Left temple  
    - Right temple
    - Nose bridge
    - Chin
    """
    
    # MediaPipe Face Mesh landmark indices for each ROI
    ROI_LANDMARKS = {
        'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        
        'left_cheek': [205, 50, 117, 118, 101, 36, 206, 216, 192, 123, 147, 213, 215, 177, 187, 207, 212, 216],
        
        'right_cheek': [425, 280, 346, 347, 330, 266, 426, 436, 416, 352, 376, 433, 435, 401, 411, 427, 432, 436],
        
        'left_temple': [103, 67, 109, 10, 338, 297, 332, 284],
        
        'right_temple': [332, 297, 338, 10, 109, 67, 103, 54],
        
        'nose_bridge': [6, 197, 195, 5, 4, 1, 19, 94, 2],
        
        'chin': [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332],
    }
    
    def __init__(self, padding: int = 5):
        """
        Initialize ROI Extractor
        
        Args:
            padding: Number of pixels to pad around each ROI
        """
        self.padding = padding
    
    def extract_roi_coordinates(
        self,
        landmarks: object,
        frame_shape: Tuple[int, int],
        roi_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract ROI coordinates from face landmarks
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: Shape of frame (height, width)
            roi_name: Optional specific ROI to extract (None for all)
        
        Returns:
            Dictionary mapping ROI names to coordinate arrays
        """
        h, w = frame_shape[:2]
        roi_coords = {}
        
        # Select which ROIs to process
        rois_to_process = {roi_name: self.ROI_LANDMARKS[roi_name]} if roi_name else self.ROI_LANDMARKS
        
        for roi_name, indices in rois_to_process.items():
            points = []
            for idx in indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append([x, y])
            
            if points:
                roi_coords[roi_name] = np.array(points, dtype=np.int32)
        
        return roi_coords
    
    def get_roi_bounding_box(self, roi_coords: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box for ROI coordinates
        
        Args:
            roi_coords: Array of (x, y) coordinates
        
        Returns:
            Tuple of (x, y, w, h)
        """
        x_min = max(0, roi_coords[:, 0].min() - self.padding)
        x_max = roi_coords[:, 0].max() + self.padding
        y_min = max(0, roi_coords[:, 1].min() - self.padding)
        y_max = roi_coords[:, 1].max() + self.padding
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def extract_roi_pixels(
        self,
        frame: np.ndarray,
        roi_coords: np.ndarray,
        use_mask: bool = True
    ) -> np.ndarray:
        """
        Extract pixel values from ROI
        
        Args:
            frame: Input frame
            roi_coords: ROI coordinates
            use_mask: If True, only extract pixels within the ROI polygon
        
        Returns:
            Extracted ROI image
        """
        # Get bounding box
        x, y, w, h = self.get_roi_bounding_box(roi_coords)
        
        # Ensure coordinates are within frame
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame))
        y = max(0, min(y, h_frame))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        
        # Extract bounding box region
        roi_image = frame[y:y+h, x:x+w].copy()
        
        if use_mask:
            # Create mask for ROI polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Adjust coordinates relative to bounding box
            adjusted_coords = roi_coords.copy()
            adjusted_coords[:, 0] -= x
            adjusted_coords[:, 1] -= y
            
            # Fill polygon
            cv2.fillPoly(mask, [adjusted_coords], 255)
            
            # Apply mask
            roi_image = cv2.bitwise_and(roi_image, roi_image, mask=mask)
        
        return roi_image
    
    def get_roi_mean_color(self, frame: np.ndarray, roi_coords: np.ndarray) -> np.ndarray:
        """
        Get mean RGB color of ROI (for rPPG signal extraction)
        
        Args:
            frame: Input frame
            roi_coords: ROI coordinates
        
        Returns:
            Array of mean [R, G, B] values
        """
        roi_image = self.extract_roi_pixels(frame, roi_coords, use_mask=True)
        
        # Calculate mean, ignoring black pixels (masked out areas)
        mask = np.any(roi_image != [0, 0, 0], axis=-1)
        
        if mask.sum() > 0:
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            mean_color = roi_rgb[mask].mean(axis=0)
            return mean_color
        else:
            return np.array([0, 0, 0])
    
    def draw_rois(
        self,
        frame: np.ndarray,
        roi_coords_dict: Dict[str, np.ndarray],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw ROIs on frame for visualization
        
        Args:
            frame: Input frame
            roi_coords_dict: Dictionary of ROI coordinates
            show_labels: Whether to show ROI labels
        
        Returns:
            Frame with drawn ROIs
        """
        annotated_frame = frame.copy()
        
        # Color palette for different ROIs
        colors = {
            'forehead': (255, 0, 0),      # Blue
            'left_cheek': (0, 255, 0),    # Green
            'right_cheek': (0, 255, 255), # Yellow
            'left_temple': (255, 0, 255), # Magenta
            'right_temple': (128, 0, 255),# Purple
            'nose_bridge': (0, 128, 255), # Orange
            'chin': (128, 255, 0),        # Lime
        }
        
        for roi_name, roi_coords in roi_coords_dict.items():
            color = colors.get(roi_name, (255, 255, 255))
            
            # Draw polygon
            cv2.polylines(annotated_frame, [roi_coords], isClosed=True, 
                         color=color, thickness=2)
            
            # Draw filled semi-transparent overlay
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [roi_coords], color)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            
            # Draw label
            if show_labels:
                # Get center of ROI
                center = roi_coords.mean(axis=0).astype(int)
                cv2.putText(annotated_frame, roi_name, tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
    
    def extract_all_rois_from_video(
        self,
        video_path: str,
        landmarks_list: List[Optional[object]],
        max_frames: Optional[int] = None
    ) -> Dict[str, List[np.ndarray]]:
        """
        Extract ROI signals from entire video
        
        Args:
            video_path: Path to video file
            landmarks_list: List of face landmarks for each frame
            max_frames: Maximum number of frames to process
        
        Returns:
            Dictionary mapping ROI names to lists of mean RGB values per frame
        """
        cap = cv2.VideoCapture(video_path)
        
        roi_signals = {roi_name: [] for roi_name in self.ROI_LANDMARKS.keys()}
        frame_count = 0
        
        for frame_idx, landmarks in enumerate(landmarks_list):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if landmarks is not None:
                # Extract ROI coordinates
                roi_coords_dict = self.extract_roi_coordinates(landmarks, frame.shape)
                
                # Get mean color for each ROI
                for roi_name, roi_coords in roi_coords_dict.items():
                    mean_color = self.get_roi_mean_color(frame, roi_coords)
                    roi_signals[roi_name].append(mean_color)
            else:
                # No face detected, append zeros
                for roi_name in roi_signals.keys():
                    roi_signals[roi_name].append(np.array([0, 0, 0]))
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        
        # Convert lists to arrays
        for roi_name in roi_signals.keys():
            roi_signals[roi_name] = np.array(roi_signals[roi_name])
        
        return roi_signals


def test_roi_extraction(video_path: str, output_path: str = "outputs/roi_extraction_test", num_frames: int = 5):
    """
    Test ROI extraction on a video
    
    Args:
        video_path: Path to test video
        output_path: Path to save annotated frames
        num_frames: Number of frames to test
    """
    try:
        from .face_detection import FaceDetector
    except ImportError:
        from face_detection import FaceDetector
    
    print(f"\n{'='*70}")
    print("TESTING ROI EXTRACTION")
    print(f"{'='*70}\n")
    
    # Initialize
    face_detector = FaceDetector()
    roi_extractor = ROIExtractor(padding=5)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    print(f"📹 Processing video: {Path(video_path).name}\n")
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_count = 0
    
    for i in range(num_frames):
        # Sample frames uniformly
        frame_idx = int((i / num_frames) * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect face
        landmarks = face_detector.detect_face(frame)
        
        if landmarks:
            # Extract ROI coordinates
            roi_coords_dict = roi_extractor.extract_roi_coordinates(landmarks, frame.shape)
            
            print(f"   Frame {frame_idx:4d}: ✅ Extracted {len(roi_coords_dict)} ROIs")
            
            # Visualize ROIs
            annotated_frame = roi_extractor.draw_rois(frame, roi_coords_dict, show_labels=True)
            
            # Save
            output_file = f"{output_path}/frame_{frame_idx:04d}_rois.jpg"
            cv2.imwrite(output_file, annotated_frame)
            
            # Print mean colors for each ROI
            for roi_name, roi_coords in roi_coords_dict.items():
                mean_color = roi_extractor.get_roi_mean_color(frame, roi_coords)
                print(f"      {roi_name:15} Mean RGB: {mean_color}")
            
            roi_count += 1
        else:
            print(f"   Frame {frame_idx:4d}: ❌ No face detected")
    
    cap.release()
    
    print(f"\n{'='*70}")
    print(f"RESULTS: Successfully extracted ROIs from {roi_count}/{num_frames} frames")
    print(f"📁 Annotated frames saved to: {output_path}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test on videos
    test_video_real = "/home/traderx/DeepFake/Celeb-DF-v2/Celeb-real/id0_0000.mp4"
    test_roi_extraction(test_video_real, num_frames=5)
    
    test_video_fake = "/home/traderx/DeepFake/Celeb-DF-v2/Celeb-synthesis/id0_id1_0000.mp4"
    test_roi_extraction(test_video_fake, num_frames=5)

