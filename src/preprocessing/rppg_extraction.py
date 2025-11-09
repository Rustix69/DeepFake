"""
rPPG Signal Extraction - Production Quality Implementation
Multiple algorithms: CHROM, POS, ICA
Designed for maximum accuracy in deepfake detection
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, detrend
from sklearn.decomposition import FastICA
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class RPPGExtractor:
    """
    High-accuracy rPPG signal extraction with multiple algorithms
    
    Implements three state-of-the-art methods:
    1. CHROM (Chrominance-based) - Best for motion robustness
    2. POS (Plane-Orthogonal-to-Skin) - Good baseline
    3. ICA (Independent Component Analysis) - Physiologically accurate
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        method: str = 'chrom',
        hr_range: Tuple[float, float] = (0.7, 3.5)  # 42-210 BPM
    ):
        """
        Initialize rPPG extractor
        
        Args:
            fps: Frames per second of video
            method: 'chrom', 'pos', or 'ica'
            hr_range: Heart rate frequency range in Hz (default: 42-210 BPM)
        """
        self.fps = fps
        self.method = method.lower()
        self.hr_range = hr_range
        
        # Validate method
        if self.method not in ['chrom', 'pos', 'ica']:
            raise ValueError(f"Method must be 'chrom', 'pos', or 'ica', got '{method}'")
        
        print(f"✅ Initialized rPPG Extractor: {self.method.upper()} method @ {fps} FPS")
    
    def extract_pulse_signal(
        self,
        rgb_signal: np.ndarray,
        apply_filtering: bool = True
    ) -> np.ndarray:
        """
        Extract pulse signal from RGB temporal data
        
        Args:
            rgb_signal: Array of shape (T, 3) with R, G, B values over time
            apply_filtering: Whether to apply bandpass filtering
        
        Returns:
            Pulse signal array of shape (T,)
        """
        if len(rgb_signal) < 30:  # Need at least 1 second of data
            print(f"⚠️  Warning: Signal too short ({len(rgb_signal)} frames)")
            return np.zeros(len(rgb_signal))
        
        # Normalize RGB signal
        rgb_normalized = self._normalize_signal(rgb_signal)
        
        # Extract pulse based on method
        if self.method == 'chrom':
            pulse_signal = self._extract_chrom(rgb_normalized)
        elif self.method == 'pos':
            pulse_signal = self._extract_pos(rgb_normalized)
        elif self.method == 'ica':
            pulse_signal = self._extract_ica(rgb_normalized)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Apply bandpass filtering
        if apply_filtering and len(pulse_signal) >= 30:
            pulse_signal = self._bandpass_filter(pulse_signal)
        
        return pulse_signal
    
    def _normalize_signal(self, rgb_signal: np.ndarray) -> np.ndarray:
        """
        Normalize RGB signal using temporal normalization
        
        Args:
            rgb_signal: Raw RGB signal (T, 3)
        
        Returns:
            Normalized signal (T, 3)
        """
        # Detrend to remove slow variations
        rgb_detrended = np.zeros_like(rgb_signal)
        for i in range(3):
            rgb_detrended[:, i] = detrend(rgb_signal[:, i])
        
        # Normalize by mean
        mean_vals = np.mean(rgb_signal, axis=0)
        mean_vals[mean_vals == 0] = 1  # Avoid division by zero
        
        rgb_normalized = rgb_detrended / mean_vals
        
        return rgb_normalized
    
    def _extract_chrom(self, rgb_normalized: np.ndarray) -> np.ndarray:
        """
        CHROM (Chrominance-based) method
        
        Best for motion robustness. Uses chrominance information.
        Reference: "Improved Motion Robustness of Remote-PPG by Using the 
                   Blood Volume Pulse Signature" (De Haan & Jeanne, 2013)
        
        Args:
            rgb_normalized: Normalized RGB signal (T, 3)
        
        Returns:
            Pulse signal (T,)
        """
        # Extract color channels
        R = rgb_normalized[:, 0]
        G = rgb_normalized[:, 1]
        B = rgb_normalized[:, 2]
        
        # CHROM algorithm
        Xs = 3 * R - 2 * G
        Ys = 1.5 * R + G - 1.5 * B
        
        # Calculate chrominance signal
        alpha = np.std(Xs) / np.std(Ys) if np.std(Ys) > 0 else 0
        pulse_signal = Xs - alpha * Ys
        
        return pulse_signal
    
    def _extract_pos(self, rgb_normalized: np.ndarray) -> np.ndarray:
        """
        POS (Plane-Orthogonal-to-Skin) method
        
        Projects RGB signals onto plane orthogonal to skin tone.
        Reference: "Algorithmic Principles of Remote PPG" (Wang et al., 2017)
        
        Args:
            rgb_normalized: Normalized RGB signal (T, 3)
        
        Returns:
            Pulse signal (T,)
        """
        # Extract color channels
        R = rgb_normalized[:, 0]
        G = rgb_normalized[:, 1]
        B = rgb_normalized[:, 2]
        
        # POS algorithm
        X = R - G
        Y = R + G - 2 * B
        
        # Calculate alpha (projection)
        alpha = np.std(X) / np.std(Y) if np.std(Y) > 0 else 0
        pulse_signal = X - alpha * Y
        
        return pulse_signal
    
    def _extract_ica(self, rgb_normalized: np.ndarray) -> np.ndarray:
        """
        ICA (Independent Component Analysis) method
        
        Separates RGB into independent components and selects pulse.
        Most physiologically accurate but sensitive to noise.
        
        Args:
            rgb_normalized: Normalized RGB signal (T, 3)
        
        Returns:
            Pulse signal (T,)
        """
        if len(rgb_normalized) < 30:
            return np.zeros(len(rgb_normalized))
        
        try:
            # Apply ICA
            ica = FastICA(n_components=3, random_state=42, max_iter=1000)
            sources = ica.fit_transform(rgb_normalized)
            
            # Select component with strongest periodicity in HR range
            best_component = 0
            best_power = 0
            
            for i in range(3):
                # Compute power spectral density
                freqs, psd = signal.periodogram(sources[:, i], fs=self.fps)
                
                # Find power in HR range
                hr_mask = (freqs >= self.hr_range[0]) & (freqs <= self.hr_range[1])
                power_in_hr_range = np.sum(psd[hr_mask])
                
                if power_in_hr_range > best_power:
                    best_power = power_in_hr_range
                    best_component = i
            
            pulse_signal = sources[:, best_component]
            
        except Exception as e:
            print(f"⚠️  ICA failed: {e}, using CHROM fallback")
            pulse_signal = self._extract_chrom(rgb_normalized)
        
        return pulse_signal
    
    def _bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter in heart rate range
        
        Args:
            signal_data: Input signal
        
        Returns:
            Filtered signal
        """
        # Design Butterworth bandpass filter
        nyquist = 0.5 * self.fps
        low = self.hr_range[0] / nyquist
        high = self.hr_range[1] / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))
        
        if low >= high:
            print("⚠️  Invalid filter range, skipping filtering")
            return signal_data
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal_data)
            return filtered_signal
        except Exception as e:
            print(f"⚠️  Filtering failed: {e}")
            return signal_data
    
    def estimate_heart_rate(self, pulse_signal: np.ndarray) -> float:
        """
        Estimate heart rate from pulse signal using FFT
        
        Args:
            pulse_signal: Pulse signal array
        
        Returns:
            Estimated heart rate in BPM
        """
        if len(pulse_signal) < 30:
            return 0.0
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(pulse_signal, fs=self.fps)
        
        # Find peak in heart rate range
        hr_mask = (freqs >= self.hr_range[0]) & (freqs <= self.hr_range[1])
        
        if not np.any(hr_mask):
            return 0.0
        
        peak_freq = freqs[hr_mask][np.argmax(psd[hr_mask])]
        heart_rate_bpm = peak_freq * 60.0  # Convert Hz to BPM
        
        return heart_rate_bpm
    
    def compute_signal_quality(self, pulse_signal: np.ndarray) -> Dict[str, float]:
        """
        Assess quality of extracted pulse signal
        
        Args:
            pulse_signal: Pulse signal array
        
        Returns:
            Dictionary with quality metrics
        """
        if len(pulse_signal) < 30:
            return {
                'snr': 0.0,
                'quality_score': 0.0,
                'peak_power': 0.0,
                'is_good_quality': False
            }
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(pulse_signal, fs=self.fps)
        
        # Calculate SNR (Signal-to-Noise Ratio)
        hr_mask = (freqs >= self.hr_range[0]) & (freqs <= self.hr_range[1])
        noise_mask = ~hr_mask
        
        signal_power = np.sum(psd[hr_mask])
        noise_power = np.sum(psd[noise_mask])
        
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0
        
        # Peak power in HR range
        peak_power = np.max(psd[hr_mask]) if np.any(hr_mask) else 0.0
        
        # Quality score (0-1 scale)
        quality_score = np.clip(snr / 20.0, 0, 1)  # Normalize SNR to 0-1
        
        # Good quality threshold: SNR > 5 dB
        is_good_quality = snr > 5.0
        
        return {
            'snr': float(snr),
            'quality_score': float(quality_score),
            'peak_power': float(peak_power),
            'is_good_quality': bool(is_good_quality)
        }
    
    def extract_from_roi_signals(
        self,
        roi_signals: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Extract pulse signals from multiple ROI signals
        
        Args:
            roi_signals: Dictionary mapping ROI names to RGB signals (T, 3)
        
        Returns:
            Dictionary with pulse signals and metrics for each ROI
        """
        results = {}
        
        for roi_name, rgb_signal in roi_signals.items():
            # Skip if signal is too short or all zeros
            if len(rgb_signal) < 30 or np.all(rgb_signal == 0):
                results[roi_name] = {
                    'pulse_signal': np.zeros(len(rgb_signal)),
                    'heart_rate_bpm': 0.0,
                    'quality': {
                        'snr': 0.0,
                        'quality_score': 0.0,
                        'peak_power': 0.0,
                        'is_good_quality': False
                    }
                }
                continue
            
            # Extract pulse signal
            pulse_signal = self.extract_pulse_signal(rgb_signal)
            
            # Estimate heart rate
            heart_rate = self.estimate_heart_rate(pulse_signal)
            
            # Assess quality
            quality = self.compute_signal_quality(pulse_signal)
            
            results[roi_name] = {
                'pulse_signal': pulse_signal,
                'heart_rate_bpm': float(heart_rate),
                'quality': quality
            }
        
        return results
    
    def analyze_cross_region_consistency(
        self,
        pulse_results: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Analyze consistency across different facial regions
        
        This is KEY for deepfake detection - real faces should have
        consistent pulse signals across regions, fakes may not.
        
        Args:
            pulse_results: Results from extract_from_roi_signals
        
        Returns:
            Dictionary with consistency metrics
        """
        # Extract heart rates and quality scores
        heart_rates = []
        quality_scores = []
        good_quality_rois = []
        
        for roi_name, result in pulse_results.items():
            if result['quality']['is_good_quality']:
                heart_rates.append(result['heart_rate_bpm'])
                quality_scores.append(result['quality']['quality_score'])
                good_quality_rois.append(roi_name)
        
        if len(heart_rates) < 2:
            return {
                'hr_std': 999.0,  # High value = inconsistent
                'hr_coefficient_of_variation': 999.0,
                'avg_quality_score': 0.0,
                'num_good_quality_rois': len(good_quality_rois),
                'consistency_score': 0.0,
                'is_consistent': False
            }
        
        # Calculate heart rate consistency
        hr_mean = np.mean(heart_rates)
        hr_std = np.std(heart_rates)
        hr_cv = hr_std / hr_mean if hr_mean > 0 else 999.0
        
        # Average quality
        avg_quality = np.mean(quality_scores)
        
        # Consistency score: low std + high quality = high consistency
        # Real faces should have HR std < 5 BPM across regions
        consistency_score = avg_quality * (1.0 / (1.0 + hr_std / 5.0))
        
        # Consistent if HR std < 10 BPM and at least 3 good ROIs
        is_consistent = (hr_std < 10.0) and (len(good_quality_rois) >= 3)
        
        return {
            'hr_std': float(hr_std),
            'hr_coefficient_of_variation': float(hr_cv),
            'avg_quality_score': float(avg_quality),
            'num_good_quality_rois': len(good_quality_rois),
            'consistency_score': float(consistency_score),
            'is_consistent': bool(is_consistent),
            'good_quality_rois': good_quality_rois,
            'heart_rates': heart_rates
        }


def compare_rppg_methods(
    rgb_signal: np.ndarray,
    fps: float = 30.0,
    method_names: List[str] = ['chrom', 'pos', 'ica']
) -> Dict[str, Dict]:
    """
    Compare different rPPG extraction methods on same signal
    
    Args:
        rgb_signal: RGB signal (T, 3)
        fps: Frames per second
        method_names: Methods to compare
    
    Returns:
        Dictionary with results from each method
    """
    results = {}
    
    for method in method_names:
        extractor = RPPGExtractor(fps=fps, method=method)
        pulse_signal = extractor.extract_pulse_signal(rgb_signal)
        heart_rate = extractor.estimate_heart_rate(pulse_signal)
        quality = extractor.compute_signal_quality(pulse_signal)
        
        results[method] = {
            'pulse_signal': pulse_signal,
            'heart_rate_bpm': heart_rate,
            'quality': quality
        }
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("rPPG EXTRACTION MODULE - TESTING")
    print("="*70 + "\n")
    
    # This will be tested with actual video data in the next module
    print("✅ rPPG extraction module loaded successfully")
    print("\nImplemented methods:")
    print("  1. CHROM (Chrominance-based) - Best for motion robustness")
    print("  2. POS (Plane-Orthogonal-to-Skin) - Good baseline")
    print("  3. ICA (Independent Component Analysis) - Physiologically accurate")
    print("\nFeatures:")
    print("  ✓ Bandpass filtering (42-210 BPM range)")
    print("  ✓ Signal quality assessment (SNR-based)")
    print("  ✓ Heart rate estimation")
    print("  ✓ Cross-region consistency analysis")
    print("\n" + "="*70)

