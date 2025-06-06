"""
Enhanced detection stabilizer using adaptive size stabilization.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .adaptive_size_stabilizer import AdaptiveSizeStabilizer


class DetectionStabilizerV2:
    """
    Stabilizes detection bounding boxes using adaptive size stabilization
    with motion-aware smoothing for better visual quality.
    """
    
    def __init__(self, 
                 smoothing_factor: float = 0.3,
                 use_adaptive_size: bool = True,
                 position_smoothing: float = 0.3,
                 size_smoothing: float = 0.1):
        """
        Initialize the detection stabilizer.
        
        Args:
            smoothing_factor: Overall smoothing factor (for compatibility)
            use_adaptive_size: Whether to use adaptive size stabilization
            position_smoothing: Smoothing for position (higher = more responsive)
            size_smoothing: Base smoothing for size (lower = more stable)
        """
        self.smoothing_factor = smoothing_factor
        self.use_adaptive_size = use_adaptive_size
        
        if use_adaptive_size:
            self.size_stabilizer = AdaptiveSizeStabilizer(
                history_window=15,
                position_smoothing=position_smoothing,
                size_smoothing_base=size_smoothing,
                motion_threshold=10.0,
                aspect_ratio_tolerance=0.2
            )
        
        # Fallback simple EMA
        self.history: Dict[int, np.ndarray] = {}
        
    def update(self, tracker_id: int, bbox: Tuple[float, float, float, float],
               confidence: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Update the bounding box for a given tracker.
        
        Args:
            tracker_id: Unique identifier for the tracked object
            bbox: Current frame bounding box (x1, y1, x2, y2)
            confidence: Detection confidence (0.0-1.0)
            
        Returns:
            Smoothed bounding box (x1, y1, x2, y2)
        """
        if self.use_adaptive_size:
            return self.size_stabilizer.update(tracker_id, bbox, confidence)
        else:
            return self._simple_smooth(tracker_id, bbox)
    
    def _simple_smooth(self, tracker_id: int, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Simple exponential moving average smoothing."""
        current_bbox = np.array(bbox, dtype=np.float32)
        
        if tracker_id not in self.history:
            self.history[tracker_id] = current_bbox
            return tuple(np.round(current_bbox).astype(int))
        
        # Apply EMA
        previous_bbox = self.history[tracker_id]
        smoothed_bbox = (self.smoothing_factor * current_bbox + 
                        (1 - self.smoothing_factor) * previous_bbox)
        
        self.history[tracker_id] = smoothed_bbox
        
        return tuple(np.round(smoothed_bbox).astype(int))
    
    def cleanup_old_trackers(self, active_tracker_ids: set):
        """Remove trackers that are no longer active."""
        if self.use_adaptive_size:
            self.size_stabilizer.cleanup_old_trackers(active_tracker_ids)
        
        # Clean up simple history
        trackers_to_remove = [tid for tid in self.history if tid not in active_tracker_ids]
        for tracker_id in trackers_to_remove:
            del self.history[tracker_id]
    
    def reset(self):
        """Clear all tracking history."""
        if self.use_adaptive_size and hasattr(self, 'size_stabilizer'):
            # Clear all histories in size stabilizer
            self.size_stabilizer.position_history.clear()
            self.size_stabilizer.size_history.clear()
            self.size_stabilizer.aspect_ratio_history.clear()
            self.size_stabilizer.velocity_history.clear()
            self.size_stabilizer.smooth_positions.clear()
            self.size_stabilizer.smooth_sizes.clear()
        
        self.history.clear()
    
    def get_history_size(self) -> int:
        """Get the number of tracked objects in history."""
        if self.use_adaptive_size:
            return len(self.size_stabilizer.position_history)
        return len(self.history)