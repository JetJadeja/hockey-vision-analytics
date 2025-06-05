"""
Detection stabilizer using Exponential Moving Average (EMA) for smooth bounding box transitions.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DetectionStabilizer:
    """
    Stabilizes detection bounding boxes using exponential moving average (EMA).
    This reduces jitter while maintaining responsive tracking of actual movement.
    """
    
    def __init__(self, smoothing_factor: float = 0.3):
        """
        Initialize the detection stabilizer.
        
        Args:
            smoothing_factor: Weight given to current frame (0.0-1.0).
                             0.3 means 70% weight on current, 30% on history.
        """
        self.smoothing_factor = smoothing_factor
        self.history: Dict[int, np.ndarray] = {}
        
    def update(self, tracker_id: int, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Update the bounding box for a given tracker using EMA smoothing.
        
        Args:
            tracker_id: Unique identifier for the tracked object
            bbox: Current frame bounding box (x1, y1, x2, y2)
            
        Returns:
            Smoothed bounding box (x1, y1, x2, y2)
        """
        current_bbox = np.array(bbox, dtype=np.float32)
        
        if tracker_id not in self.history:
            # First time seeing this tracker, no smoothing possible
            self.history[tracker_id] = current_bbox
            return tuple(current_bbox)
        
        # Apply exponential moving average
        # EMA = α * current + (1 - α) * previous
        # where α is the smoothing factor
        previous_bbox = self.history[tracker_id]
        smoothed_bbox = (self.smoothing_factor * current_bbox + 
                        (1 - self.smoothing_factor) * previous_bbox)
        
        # Update history with smoothed values (keep as floats for accuracy)
        self.history[tracker_id] = smoothed_bbox
        
        # Round to nearest pixel for stable rendering
        rounded_bbox = np.round(smoothed_bbox).astype(int)
        
        # Return as tuple of integers
        return tuple(rounded_bbox)
    
    def cleanup_old_trackers(self, active_tracker_ids: set):
        """
        Remove trackers that are no longer active to prevent memory buildup.
        
        Args:
            active_tracker_ids: Set of currently active tracker IDs
        """
        # Find trackers to remove
        trackers_to_remove = []
        for tracker_id in self.history:
            if tracker_id not in active_tracker_ids:
                trackers_to_remove.append(tracker_id)
        
        # Remove inactive trackers
        for tracker_id in trackers_to_remove:
            del self.history[tracker_id]
    
    def reset(self):
        """Clear all tracking history."""
        self.history.clear()
    
    def get_history_size(self) -> int:
        """Get the number of tracked objects in history."""
        return len(self.history)