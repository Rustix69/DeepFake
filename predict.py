"""
Deepfake Detection - Inference Script
Use trained model to predict if a video is real or fake
"""

import sys
sys.path.append('src')

import torch
import cv2
import argparse
from pathlib import Path
import numpy as np

from models.deepfake_detector import DeepfakeDetector
from preprocessing.face_detection import FaceDetector
from preprocessing.roi_extraction import ROIExtractor
from preprocessing.rppg_extraction import RPPGExtractor
from preprocessing.feature_extraction import DeepfakeFeatureExtractor


class DeepfakePredictor:
    """
    Easy-to-use predictor for deepfake detection
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize predictor with trained model
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize preprocessing modules
        print("📦 Loading preprocessing modules...")
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.rppg_extractor = RPPGExtractor(method='chrom')
        self.feature_extractor = DeepfakeFeatureExtractor()
        
        # Load trained model
        print(f"🔄 Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        print("✅ Model loaded successfully!\n")
    
    def _load_model(self, checkpoint_path: str) -> DeepfakeDetector:
        """Load trained model from checkpoint"""
        # Initialize model architecture
        model = DeepfakeDetector(
            sequence_length=150,
            input_channels=3,
            num_regions=7,
            temporal_feature_dim=256,
            transformer_dim=256,
            num_transformer_layers=4,
            num_heads=8,
            fusion_hidden_dim=512,
            handcrafted_dim=49,
            use_handcrafted=True,
            dropout=0.3,
            num_classes=2
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_video(
        self, 
        video_path: str,
        num_frames: int = 150,
        threshold: float = 0.5
    ) -> dict:
        """
        Predict if a video is real or fake
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze
            threshold: Classification threshold (0.5 = balanced)
        
        Returns:
            dict with:
                - prediction: 'REAL' or 'FAKE'
                - confidence: 0-100%
                - fake_probability: 0-1
                - details: additional info
        """
        print(f"\n{'='*60}")
        print(f"🎥 Analyzing: {Path(video_path).name}")
        print(f"{'='*60}\n")
        
        # Step 1: Process video
        print("1️⃣  Processing video...")
        processed_data = self._process_video(video_path, num_frames)
        
        if processed_data is None:
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'fake_probability': 0.0,
                'details': 'Failed to process video (no face detected or processing error)'
            }
        
        # Step 2: Make prediction
        print("2️⃣  Running model inference...")
        with torch.no_grad():
            # Prepare inputs
            rppg_signals = processed_data['rppg_signals'].unsqueeze(0).to(self.device)
            handcrafted = processed_data['handcrafted_features'].unsqueeze(0).to(self.device)
            
            # Forward pass
            output = self.model(rppg_signals, handcrafted)
            
            # Get results
            fake_prob = output['probabilities'][0, 1].item()  # Probability of fake
            prediction = 'FAKE' if fake_prob > threshold else 'REAL'
            confidence = fake_prob if prediction == 'FAKE' else (1 - fake_prob)
        
        # Step 3: Return results
        print(f"\n{'='*60}")
        print(f"🎯 RESULT: {prediction}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print(f"📈 Fake Probability: {fake_prob*100:.2f}%")
        print(f"{'='*60}\n")
        
        return {
            'prediction': prediction,
            'confidence': confidence * 100,
            'fake_probability': fake_prob,
            'details': {
                'num_frames_processed': processed_data['num_frames'],
                'face_detection_rate': processed_data['face_detection_rate'],
                'avg_heart_rate': processed_data.get('avg_heart_rate', 'N/A'),
                'signal_quality': processed_data.get('signal_quality', 'N/A')
            }
        }
    
    def _process_video(self, video_path: str, num_frames: int) -> dict:
        """Process video through preprocessing pipeline"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames_data = []
            faces_detected = 0
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Detect face
                result = self.face_detector.detect_face(frame)
                if result is None:
                    continue
                
                faces_detected += 1
                
                # Extract ROIs
                rois = self.roi_extractor.extract_rois(frame, result['landmarks'])
                if rois is None:
                    continue
                
                frames_data.append(rois)
            
            cap.release()
            
            # Check if enough faces detected
            if faces_detected < num_frames * 0.5:  # At least 50%
                print(f"⚠️  Warning: Only {faces_detected}/{num_frames} faces detected")
                return None
            
            # Pad or truncate to exact length
            if len(frames_data) < num_frames:
                # Pad with last frame
                while len(frames_data) < num_frames:
                    frames_data.append(frames_data[-1])
            else:
                frames_data = frames_data[:num_frames]
            
            # Convert to numpy array: (T, ROI, C)
            rppg_array = np.array(frames_data)  # (T, 7, 3)
            
            # Transpose to (ROI, C, T) for model input
            rppg_tensor = torch.FloatTensor(rppg_array).permute(1, 2, 0)  # (7, 3, 150)
            
            # Extract handcrafted features
            handcrafted_features = self.feature_extractor.extract_features(rppg_array)
            handcrafted_tensor = torch.FloatTensor(handcrafted_features)
            
            return {
                'rppg_signals': rppg_tensor,
                'handcrafted_features': handcrafted_tensor,
                'num_frames': len(frames_data),
                'face_detection_rate': faces_detected / num_frames
            }
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Detect deepfakes in videos")
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (0-1)')
    parser.add_argument('--num-frames', type=int, default=150,
                       help='Number of frames to analyze')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video_path).exists():
        print(f"❌ Error: Video not found: {args.video_path}")
        return
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        print(f"\nPlease train the model first:")
        print(f"  python train.py")
        return
    
    # Initialize predictor
    predictor = DeepfakePredictor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Make prediction
    result = predictor.predict_video(
        video_path=args.video_path,
        num_frames=args.num_frames,
        threshold=args.threshold
    )
    
    # Save result
    print("Results:", result)


if __name__ == "__main__":
    main()
