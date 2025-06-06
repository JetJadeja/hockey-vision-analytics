"""
Advanced detection stabilizer using Kalman filtering for smooth bounding box transitions.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .kalman_tracker import KalmanBoxFilter


class DetectionStabilizer:
    """
    Stabilizes detection bounding boxes using Kalman filtering with adaptive smoothing.
    This provides smooth, predictive tracking that adapts to motion patterns.
    """
    
    def __init__(self, smoothing_factor: float = 0.3, use_kalman: bool = True,
                 velocity_threshold: float = 10.0, size_stability_factor: float = 0.3):
        """
        Initialize the detection stabilizer.
        
        Args:
            smoothing_factor: Base smoothing factor (0.0-1.0)
            use_kalman: Whether to use Kalman filtering (fallback to EMA if False)
            velocity_threshold: Pixel/frame threshold for high velocity
            size_stability_factor: How much to stabilize box size (0.0-1.0)
        """
        self.smoothing_factor = smoothing_factor
        self.use_kalman = use_kalman
        self.velocity_threshold = velocity_threshold
        self.size_stability_factor = size_stability_factor
        
        # Kalman filters for each tracker
        self.kalman_filters: Dict[int, KalmanBoxFilter] = {}
        
        # Fallback EMA history
        self.history: Dict[int, np.ndarray] = {}
        
        # Size history for stability
        self.size_history: Dict[int, list] = {}
        self.size_history_window = 5
        
    def update(self, tracker_id: int, bbox: Tuple[float, float, float, float],
               confidence: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Update the bounding box for a given tracker using adaptive smoothing.
        
        Args:
            tracker_id: Unique identifier for the tracked object
            bbox: Current frame bounding box (x1, y1, x2, y2)
            confidence: Detection confidence (0.0-1.0)
            
        Returns:
            Smoothed bounding box (x1, y1, x2, y2)
        """
        if self.use_kalman:
            return self._update_kalman(tracker_id, bbox, confidence)
        else:
            return self._update_ema(tracker_id, bbox)
    
    def _update_kalman(self, tracker_id: int, bbox: Tuple[float, float, float, float],
                       confidence: float) -> Tuple[float, float, float, float]:
        """Update using Kalman filter with adaptive smoothing."""
        # Initialize Kalman filter if new tracker
        if tracker_id not in self.kalman_filters:
            self.kalman_filters[tracker_id] = KalmanBoxFilter(bbox)
            self.size_history[tracker_id] = []
            return self._round_bbox(bbox)
        
        kf = self.kalman_filters[tracker_id]
        
        # Predict next state
        kf.predict()
        
        # Get predicted bbox before update
        predicted_bbox = kf.get_state_bbox()
        
        # Calculate adaptive smoothing factor
        motion_magnitude = kf.get_motion_magnitude()
        
        # Reduce smoothing for fast motion
        if motion_magnitude > self.velocity_threshold:
            motion_factor = min(motion_magnitude / (self.velocity_threshold * 2), 1.0)
            adaptive_smoothing = self.smoothing_factor * (1 - motion_factor * 0.7)
        else:
            adaptive_smoothing = self.smoothing_factor
        
        # Further adjust based on confidence
        adaptive_smoothing *= (2.0 - confidence)  # Less smoothing for high confidence
        adaptive_smoothing = np.clip(adaptive_smoothing, 0.1, 0.9)
        
        # Blend measurement with prediction
        blended_bbox = self._blend_bboxes(predicted_bbox, bbox, adaptive_smoothing)
        
        # Update Kalman filter
        kf.update(blended_bbox, confidence)
        
        # Get final smoothed bbox
        smoothed_bbox = kf.get_state_bbox()
        
        # Apply size stabilization
        smoothed_bbox = self._stabilize_size(tracker_id, smoothed_bbox)
        
        return self._round_bbox(smoothed_bbox)
    
    def _update_ema(self, tracker_id: int, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Fallback EMA update."""
        current_bbox = np.array(bbox, dtype=np.float32)
        
        if tracker_id not in self.history:
            self.history[tracker_id] = current_bbox
            return self._round_bbox(bbox)
        
        previous_bbox = self.history[tracker_id]
        smoothed_bbox = (self.smoothing_factor * current_bbox + 
                        (1 - self.smoothing_factor) * previous_bbox)
        
        self.history[tracker_id] = smoothed_bbox
        
        return self._round_bbox(tuple(smoothed_bbox))
    
    def _blend_bboxes(self, bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float],
                      alpha: float) -> Tuple[float, float, float, float]:
        """Blend two bboxes with given weight."""
        bbox1_arr = np.array(bbox1, dtype=np.float32)
        bbox2_arr = np.array(bbox2, dtype=np.float32)
        blended = (1 - alpha) * bbox1_arr + alpha * bbox2_arr
        return tuple(blended)
    
    def _stabilize_size(self, tracker_id: int, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Stabilize bbox size to reduce jitter."""
        x1, y1, x2, y2 = bbox
        current_w = x2 - x1
        current_h = y2 - y1
        
        # Update size history
        if tracker_id not in self.size_history:
            self.size_history[tracker_id] = []
        
        self.size_history[tracker_id].append((current_w, current_h))
        if len(self.size_history[tracker_id]) > self.size_history_window:
            self.size_history[tracker_id].pop(0)
        
        # Apply size stabilization if we have history
        if len(self.size_history[tracker_id]) >= 3:
            sizes = np.array(self.size_history[tracker_id])
            median_w = np.median(sizes[:, 0])
            median_h = np.median(sizes[:, 1])
            
            # Only stabilize if size change is small
            if (abs(current_w - median_w) / median_w < 0.15 and 
                abs(current_h - median_h) / median_h < 0.15):
                # Smooth transition
                stable_w = current_w * (1 - self.size_stability_factor) + median_w * self.size_stability_factor
                stable_h = current_h * (1 - self.size_stability_factor) + median_h * self.size_stability_factor
                
                # Reconstruct bbox with stable size
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                return (cx - stable_w/2, cy - stable_h/2, cx + stable_w/2, cy + stable_h/2)
        
        return bbox
    
    def _round_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """Smart rounding to reduce jitter."""
        result = []
        for val in bbox:
            # Hysteresis rounding - only change integer part if fractional part is extreme
            int_part = int(val)
            frac_part = val - int_part
            
            if frac_part > 0.8:
                result.append(int_part + 1)
            elif frac_part < 0.2:
                result.append(int_part)
            else:
                # Standard rounding in middle zone
                result.append(int(np.round(val)))
        
        return tuple(result)
    
    def cleanup_old_trackers(self, active_tracker_ids: set):
        """
        Remove trackers that are no longer active to prevent memory buildup.
        
        Args:
            active_tracker_ids: Set of currently active tracker IDs
        """
        # Find trackers to remove
        if self.use_kalman:
            trackers_to_remove = [tid for tid in self.kalman_filters if tid not in active_tracker_ids]
            for tracker_id in trackers_to_remove:
                del self.kalman_filters[tracker_id]
                if tracker_id in self.size_history:
                    del self.size_history[tracker_id]
        
        # Clean up EMA history
        trackers_to_remove = [tid for tid in self.history if tid not in active_tracker_ids]
        for tracker_id in trackers_to_remove:
            del self.history[tracker_id]
    
    def reset(self):
        """Clear all tracking history."""
        self.kalman_filters.clear()
        self.history.clear()
        self.size_history.clear()
    
    def get_history_size(self) -> int:
        """Get the number of tracked objects in history."""
        if self.use_kalman:
            return len(self.kalman_filters)
        return len(self.history)