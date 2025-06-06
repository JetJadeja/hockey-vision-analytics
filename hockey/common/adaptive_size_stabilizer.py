"""
Adaptive size stabilizer for smooth bounding box dimensions.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque
import scipy.stats


class AdaptiveSizeStabilizer:
    """
    Stabilizes bounding box sizes with motion-aware adaptive smoothing.
    Separates position and size tracking for better visual stability.
    """
    
    def __init__(self, 
                 history_window: int = 15,
                 position_smoothing: float = 0.3,
                 size_smoothing_base: float = 0.1,
                 motion_threshold: float = 10.0,
                 aspect_ratio_tolerance: float = 0.2):
        """
        Initialize the adaptive size stabilizer.
        
        Args:
            history_window: Number of frames to keep in history
            position_smoothing: Smoothing factor for position (higher = more responsive)
            size_smoothing_base: Base smoothing factor for size (lower = more stable)
            motion_threshold: Pixel threshold to detect fast motion
            aspect_ratio_tolerance: Allowed deviation from median aspect ratio
        """
        self.history_window = history_window
        self.position_smoothing = position_smoothing
        self.size_smoothing_base = size_smoothing_base
        self.motion_threshold = motion_threshold
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        
        # Track history for each detection
        self.position_history: Dict[int, deque] = {}
        self.size_history: Dict[int, deque] = {}
        self.aspect_ratio_history: Dict[int, deque] = {}
        self.velocity_history: Dict[int, deque] = {}
        
        # Current smooth states
        self.smooth_positions: Dict[int, np.ndarray] = {}
        self.smooth_sizes: Dict[int, np.ndarray] = {}
        
    def update(self, tracker_id: int, bbox: Tuple[float, float, float, float],
               confidence: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Update and stabilize bounding box.
        
        Args:
            tracker_id: Unique tracker identifier
            bbox: Current bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            
        Returns:
            Stabilized bounding box (x1, y1, x2, y2)
        """
        # Extract position and size
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        # Initialize if new tracker
        if tracker_id not in self.position_history:
            self._initialize_tracker(tracker_id, cx, cy, w, h)
            return bbox
        
        # Calculate motion velocity
        prev_cx, prev_cy = self.smooth_positions[tracker_id]
        velocity = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        self.velocity_history[tracker_id].append(velocity)
        
        # Update histories
        self.position_history[tracker_id].append((cx, cy))
        self.size_history[tracker_id].append((w, h))
        self.aspect_ratio_history[tracker_id].append(w / max(h, 1))
        
        # Stabilize position (more responsive)
        smooth_cx, smooth_cy = self._smooth_position(tracker_id, cx, cy, confidence)
        
        # Stabilize size (more stable, motion-aware)
        smooth_w, smooth_h = self._smooth_size(tracker_id, w, h, velocity, confidence)
        
        # Apply aspect ratio constraints
        smooth_w, smooth_h = self._constrain_aspect_ratio(tracker_id, smooth_w, smooth_h)
        
        # Update smooth states
        self.smooth_positions[tracker_id] = np.array([smooth_cx, smooth_cy])
        self.smooth_sizes[tracker_id] = np.array([smooth_w, smooth_h])
        
        # Reconstruct bbox
        return (
            smooth_cx - smooth_w / 2.0,
            smooth_cy - smooth_h / 2.0,
            smooth_cx + smooth_w / 2.0,
            smooth_cy + smooth_h / 2.0
        )
    
    def _initialize_tracker(self, tracker_id: int, cx: float, cy: float, w: float, h: float):
        """Initialize a new tracker."""
        self.position_history[tracker_id] = deque(maxlen=self.history_window)
        self.size_history[tracker_id] = deque(maxlen=self.history_window)
        self.aspect_ratio_history[tracker_id] = deque(maxlen=self.history_window)
        self.velocity_history[tracker_id] = deque(maxlen=self.history_window)
        
        self.position_history[tracker_id].append((cx, cy))
        self.size_history[tracker_id].append((w, h))
        self.aspect_ratio_history[tracker_id].append(w / max(h, 1))
        self.velocity_history[tracker_id].append(0.0)
        
        self.smooth_positions[tracker_id] = np.array([cx, cy])
        self.smooth_sizes[tracker_id] = np.array([w, h])
    
    def _smooth_position(self, tracker_id: int, cx: float, cy: float, confidence: float) -> Tuple[float, float]:
        """Smooth position with standard exponential smoothing."""
        prev_cx, prev_cy = self.smooth_positions[tracker_id]
        
        # Adjust smoothing based on confidence
        alpha = self.position_smoothing * confidence
        
        smooth_cx = alpha * cx + (1 - alpha) * prev_cx
        smooth_cy = alpha * cy + (1 - alpha) * prev_cy
        
        return smooth_cx, smooth_cy
    
    def _smooth_size(self, tracker_id: int, w: float, h: float, velocity: float, confidence: float) -> Tuple[float, float]:
        """
        Smooth size with motion-aware adaptive smoothing.
        Less smoothing when moving fast (allows natural size changes).
        More smoothing when stationary (reduces jitter).
        """
        prev_w, prev_h = self.smooth_sizes[tracker_id]
        
        # Calculate adaptive smoothing factor
        # High velocity -> higher alpha (more responsive)
        # Low velocity -> lower alpha (more stable)
        motion_factor = min(velocity / self.motion_threshold, 1.0)
        adaptive_alpha = self.size_smoothing_base + motion_factor * 0.2
        
        # Further adjust by confidence
        adaptive_alpha *= confidence
        
        # Apply percentile-based constraints
        if len(self.size_history[tracker_id]) >= 5:
            sizes = np.array(list(self.size_history[tracker_id]))
            w_25, w_75 = np.percentile(sizes[:, 0], [25, 75])
            h_25, h_75 = np.percentile(sizes[:, 1], [25, 75])
            
            # If current size is within stable range, use stronger smoothing
            if w_25 <= w <= w_75 and h_25 <= h <= h_75:
                adaptive_alpha *= 0.5
        
        # Smooth the size
        smooth_w = adaptive_alpha * w + (1 - adaptive_alpha) * prev_w
        smooth_h = adaptive_alpha * h + (1 - adaptive_alpha) * prev_h
        
        return smooth_w, smooth_h
    
    def _constrain_aspect_ratio(self, tracker_id: int, w: float, h: float) -> Tuple[float, float]:
        """
        Constrain aspect ratio to prevent unrealistic proportions.
        """
        if len(self.aspect_ratio_history[tracker_id]) < 5:
            return w, h
        
        # Get median aspect ratio
        median_ar = np.median(list(self.aspect_ratio_history[tracker_id]))
        current_ar = w / max(h, 1)
        
        # Check if current aspect ratio deviates too much
        ar_deviation = abs(current_ar - median_ar) / median_ar
        
        if ar_deviation > self.aspect_ratio_tolerance:
            # Adjust dimensions to maintain median aspect ratio
            # Preserve area while fixing aspect ratio
            area = w * h
            new_h = np.sqrt(area / median_ar)
            new_w = median_ar * new_h
            
            # Blend with original to avoid sudden jumps
            blend_factor = 0.7
            w = blend_factor * new_w + (1 - blend_factor) * w
            h = blend_factor * new_h + (1 - blend_factor) * h
        
        return w, h
    
    def cleanup_tracker(self, tracker_id: int):
        """Remove tracker data."""
        for history_dict in [self.position_history, self.size_history, 
                           self.aspect_ratio_history, self.velocity_history,
                           self.smooth_positions, self.smooth_sizes]:
            if tracker_id in history_dict:
                del history_dict[tracker_id]
    
    def cleanup_old_trackers(self, active_ids: set):
        """Remove inactive trackers."""
        all_ids = set(self.position_history.keys())
        inactive_ids = all_ids - active_ids
        for tracker_id in inactive_ids:
            self.cleanup_tracker(tracker_id)