"""
Feature Extraction for Deepfake Detection
Combines rPPG signals with spatial and temporal features
Optimized for maximum accuracy
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DeepfakeFeatureExtractor:
    """
    Extract comprehensive features for deepfake detection
    
    Combines:
    1. rPPG signal features (temporal)
    2. Cross-region consistency features
    3. Frequency domain features
    4. Statistical features
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize feature extractor
        
        Args:
            fps: Video frame rate
        """
        self.fps = fps
        self.hr_range = (0.7, 3.5)  # 42-210 BPM in Hz
    
    def extract_temporal_features(self, pulse_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal domain features from pulse signal
        
        Args:
            pulse_signal: Pulse signal array
        
        Returns:
            Dictionary of temporal features
        """
        if len(pulse_signal) < 30:
            return self._empty_temporal_features()
        
        # Basic statistics
        mean_val = np.mean(pulse_signal)
        std_val = np.std(pulse_signal)
        skewness = stats.skew(pulse_signal)
        kurtosis = stats.kurtosis(pulse_signal)
        
        # Peak detection
        peaks, properties = signal.find_peaks(pulse_signal, distance=int(self.fps * 0.3))
        num_peaks = len(peaks)
        
        # Peak-to-peak intervals (heart rate variability)
        if num_peaks > 1:
            peak_intervals = np.diff(peaks) / self.fps
            hrv_mean = np.mean(peak_intervals)
            hrv_std = np.std(peak_intervals)
            hrv_cv = hrv_std / hrv_mean if hrv_mean > 0 else 0
        else:
            hrv_mean, hrv_std, hrv_cv = 0, 0, 0
        
        # Signal energy
        energy = np.sum(pulse_signal ** 2)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(pulse_signal)) != 0)
        zcr = zero_crossings / len(pulse_signal)
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'num_peaks': int(num_peaks),
            'hrv_mean': float(hrv_mean),
            'hrv_std': float(hrv_std),
            'hrv_cv': float(hrv_cv),
            'energy': float(energy),
            'zero_crossing_rate': float(zcr)
        }
    
    def extract_frequency_features(self, pulse_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from pulse signal
        
        Args:
            pulse_signal: Pulse signal array
        
        Returns:
            Dictionary of frequency features
        """
        if len(pulse_signal) < 30:
            return self._empty_frequency_features()
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(pulse_signal, fs=self.fps)
        
        # HR range power
        hr_mask = (freqs >= self.hr_range[0]) & (freqs <= self.hr_range[1])
        total_power = np.sum(psd)
        hr_power = np.sum(psd[hr_mask])
        hr_power_ratio = hr_power / total_power if total_power > 0 else 0
        
        # Peak frequency and heart rate
        if np.any(hr_mask):
            peak_freq = freqs[hr_mask][np.argmax(psd[hr_mask])]
            peak_hr_bpm = peak_freq * 60.0
            peak_power = np.max(psd[hr_mask])
        else:
            peak_freq = 0
            peak_hr_bpm = 0
            peak_power = 0
        
        # Spectral entropy (measure of signal randomness)
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Dominant frequency ratio (peak / total in HR range)
        dominant_ratio = peak_power / hr_power if hr_power > 0 else 0
        
        return {
            'peak_frequency': float(peak_freq),
            'peak_hr_bpm': float(peak_hr_bpm),
            'hr_power': float(hr_power),
            'hr_power_ratio': float(hr_power_ratio),
            'spectral_entropy': float(spectral_entropy),
            'dominant_ratio': float(dominant_ratio),
            'peak_power': float(peak_power)
        }
    
    def extract_roi_features(self, pulse_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Extract features from all ROIs
        
        Args:
            pulse_results: Results from RPPGExtractor.extract_from_roi_signals
        
        Returns:
            Dictionary of ROI-based features
        """
        # Collect heart rates and quality scores from all ROIs
        heart_rates = []
        quality_scores = []
        snr_values = []
        
        for roi_name, result in pulse_results.items():
            hr = result['heart_rate_bpm']
            quality = result['quality']
            
            if quality['is_good_quality']:
                heart_rates.append(hr)
                quality_scores.append(quality['quality_score'])
                snr_values.append(quality['snr'])
        
        # If no good quality ROIs, return zeros
        if len(heart_rates) == 0:
            return self._empty_roi_features()
        
        # Heart rate statistics across ROIs
        hr_mean = np.mean(heart_rates)
        hr_std = np.std(heart_rates)
        hr_min = np.min(heart_rates)
        hr_max = np.max(heart_rates)
        hr_range = hr_max - hr_min
        hr_cv = hr_std / hr_mean if hr_mean > 0 else 999
        
        # Quality statistics
        avg_quality = np.mean(quality_scores)
        avg_snr = np.mean(snr_values)
        
        # Pairwise HR differences (consistency measure)
        hr_array = np.array(heart_rates)
        pairwise_diffs = []
        for i in range(len(hr_array)):
            for j in range(i+1, len(hr_array)):
                pairwise_diffs.append(abs(hr_array[i] - hr_array[j]))
        
        if pairwise_diffs:
            avg_pairwise_diff = np.mean(pairwise_diffs)
            max_pairwise_diff = np.max(pairwise_diffs)
        else:
            avg_pairwise_diff = 0
            max_pairwise_diff = 0
        
        return {
            'num_good_rois': len(heart_rates),
            'hr_mean': float(hr_mean),
            'hr_std': float(hr_std),
            'hr_min': float(hr_min),
            'hr_max': float(hr_max),
            'hr_range': float(hr_range),
            'hr_cv': float(hr_cv),
            'avg_quality_score': float(avg_quality),
            'avg_snr': float(avg_snr),
            'avg_pairwise_hr_diff': float(avg_pairwise_diff),
            'max_pairwise_hr_diff': float(max_pairwise_diff)
        }
    
    def extract_all_features(
        self,
        pulse_results: Dict[str, Dict],
        include_per_roi: bool = False
    ) -> Dict[str, float]:
        """
        Extract complete feature set for deepfake detection
        
        Args:
            pulse_results: Results from RPPGExtractor.extract_from_roi_signals
            include_per_roi: Whether to include individual ROI features
        
        Returns:
            Dictionary with all features
        """
        features = {}
        
        # Extract ROI-level features (cross-region consistency)
        roi_features = self.extract_roi_features(pulse_results)
        features.update({f'roi_{k}': v for k, v in roi_features.items()})
        
        # Extract per-ROI features (optional)
        if include_per_roi:
            for roi_name, result in pulse_results.items():
                pulse_signal = result['pulse_signal']
                
                # Temporal features
                temporal_feats = self.extract_temporal_features(pulse_signal)
                features.update({f'{roi_name}_temp_{k}': v for k, v in temporal_feats.items()})
                
                # Frequency features
                freq_feats = self.extract_frequency_features(pulse_signal)
                features.update({f'{roi_name}_freq_{k}': v for k, v in freq_feats.items()})
        
        # Otherwise, aggregate across ROIs
        else:
            # Get average temporal and frequency features across good ROIs
            all_temporal = []
            all_frequency = []
            
            for roi_name, result in pulse_results.items():
                if result['quality']['is_good_quality']:
                    pulse_signal = result['pulse_signal']
                    all_temporal.append(self.extract_temporal_features(pulse_signal))
                    all_frequency.append(self.extract_frequency_features(pulse_signal))
            
            # Average temporal features
            if all_temporal:
                for key in all_temporal[0].keys():
                    values = [f[key] for f in all_temporal]
                    features[f'avg_temp_{key}'] = float(np.mean(values))
                    features[f'std_temp_{key}'] = float(np.std(values))
            
            # Average frequency features
            if all_frequency:
                for key in all_frequency[0].keys():
                    values = [f[key] for f in all_frequency]
                    features[f'avg_freq_{key}'] = float(np.mean(values))
                    features[f'std_freq_{key}'] = float(np.std(values))
        
        return features
    
    def _empty_temporal_features(self) -> Dict[str, float]:
        """Return zero-filled temporal features"""
        return {
            'mean': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0,
            'num_peaks': 0, 'hrv_mean': 0.0, 'hrv_std': 0.0, 'hrv_cv': 0.0,
            'energy': 0.0, 'zero_crossing_rate': 0.0
        }
    
    def _empty_frequency_features(self) -> Dict[str, float]:
        """Return zero-filled frequency features"""
        return {
            'peak_frequency': 0.0, 'peak_hr_bpm': 0.0, 'hr_power': 0.0,
            'hr_power_ratio': 0.0, 'spectral_entropy': 0.0,
            'dominant_ratio': 0.0, 'peak_power': 0.0
        }
    
    def _empty_roi_features(self) -> Dict[str, float]:
        """Return zero-filled ROI features"""
        return {
            'num_good_rois': 0, 'hr_mean': 0.0, 'hr_std': 999.0,
            'hr_min': 0.0, 'hr_max': 0.0, 'hr_range': 0.0, 'hr_cv': 999.0,
            'avg_quality_score': 0.0, 'avg_snr': 0.0,
            'avg_pairwise_hr_diff': 999.0, 'max_pairwise_hr_diff': 999.0
        }


def compute_deepfake_score(features: Dict[str, float]) -> Tuple[float, str]:
    """
    Compute simple rule-based deepfake score from features
    
    This is a baseline before ML model. Real videos should have:
    - Low HR std across ROIs (< 10 BPM)
    - High number of good quality ROIs (>= 5)
    - Low HR coefficient of variation (< 0.15)
    - High average quality score (> 0.6)
    
    Args:
        features: Feature dictionary from extract_all_features
    
    Returns:
        (score, prediction) where score in [0, 1] (0=real, 1=fake)
    """
    score = 0.0
    weights = []
    
    # HR standard deviation (lower = more real)
    if 'roi_hr_std' in features:
        hr_std = features['roi_hr_std']
        hr_std_score = np.clip(hr_std / 20.0, 0, 1)  # Normalize
        score += hr_std_score * 0.3
        weights.append(0.3)
    
    # Number of good ROIs (higher = more real)
    if 'roi_num_good_rois' in features:
        num_good = features['roi_num_good_rois']
        roi_score = 1.0 - (num_good / 7.0)  # Invert: fewer good = more fake
        score += roi_score * 0.2
        weights.append(0.2)
    
    # HR coefficient of variation (higher = more fake)
    if 'roi_hr_cv' in features:
        hr_cv = features['roi_hr_cv']
        cv_score = np.clip(hr_cv / 0.3, 0, 1)
        score += cv_score * 0.2
        weights.append(0.2)
    
    # Average quality (lower = more fake)
    if 'roi_avg_quality_score' in features:
        quality = features['roi_avg_quality_score']
        quality_score = 1.0 - quality
        score += quality_score * 0.15
        weights.append(0.15)
    
    # Pairwise HR differences (higher = more fake)
    if 'roi_avg_pairwise_hr_diff' in features:
        pairwise_diff = features['roi_avg_pairwise_hr_diff']
        diff_score = np.clip(pairwise_diff / 30.0, 0, 1)
        score += diff_score * 0.15
        weights.append(0.15)
    
    # Normalize score
    if sum(weights) > 0:
        score = score / sum(weights)
    
    # Threshold at 0.5
    prediction = 'FAKE' if score > 0.5 else 'REAL'
    
    return float(score), prediction


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEATURE EXTRACTION MODULE - TESTING")
    print("="*70 + "\n")
    
    print("✅ Feature extraction module loaded successfully")
    print("\nFeature categories:")
    print("  1. Temporal features (10 per ROI)")
    print("  2. Frequency features (7 per ROI)")
    print("  3. Cross-ROI consistency features (11 features)")
    print("  4. Aggregated statistics")
    print("\nTotal: ~40-50 features for ML model")
    print("\nAlso includes rule-based baseline detector")
    print("="*70 + "\n")

