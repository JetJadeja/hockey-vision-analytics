import numpy as np
import supervision as sv
from typing import Dict, Optional, Union
from .detection_stabilizer import DetectionStabilizer


class SmoothAnnotator:
    """
    Wrapper for supervision annotators that applies smoothing to bounding boxes
    only for visualization purposes, without affecting the underlying detections.
    """
    
    def __init__(self, annotator: Union[sv.BoxAnnotator, sv.EllipseAnnotator], 
                 smoothing_factor: float = 0.3, use_kalman: bool = True):
        """
        Initialize the smooth annotator wrapper.
        
        Args:
            annotator: The base supervision annotator to wrap
            smoothing_factor: How much to smooth (0=no smoothing, 1=full smoothing)
            use_kalman: Whether to use Kalman filtering (more advanced smoothing)
        """
        self.annotator = annotator
        self.stabilizer = DetectionStabilizer(
            smoothing_factor=smoothing_factor,
            use_kalman=use_kalman,
            velocity_threshold=15.0,  # Adjusted for hockey's fast pace
            size_stability_factor=0.4  # More size stability
        )
    
    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        labels: Optional[list] = None,
        custom_color_lookup: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Annotate the scene with smoothed bounding boxes.
        
        Args:
            scene: The frame to annotate
            detections: Detection results (not modified)
            labels: Optional labels for detections
            custom_color_lookup: Optional custom color array
            
        Returns:
            Annotated frame
        """
        if detections.tracker_id is None or len(detections) == 0:
            # No tracking or no detections, use original annotator
            if labels is not None:
                return self.annotator.annotate(scene, detections, labels=labels, custom_color_lookup=custom_color_lookup)
            else:
                return self.annotator.annotate(scene, detections, custom_color_lookup=custom_color_lookup)
        
        # Create a copy of detections for annotation only
        smoothed_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            mask=detections.mask,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=detections.tracker_id,
            data=detections.data if hasattr(detections, 'data') else {}
        )
        
        # Apply smoothing to each detection individually
        for i, (tracker_id, bbox) in enumerate(zip(detections.tracker_id, detections.xyxy)):
            # Get confidence if available
            confidence = detections.confidence[i] if detections.confidence is not None else 1.0
            smoothed_bbox = self.stabilizer.update(tracker_id, tuple(bbox), confidence)
            smoothed_detections.xyxy[i] = np.array(smoothed_bbox)
        
        # Clean up old trackers
        active_ids = set(detections.tracker_id.tolist())
        self.stabilizer.cleanup_old_trackers(active_ids)
        
        # Use the base annotator with smoothed detections
        if labels is not None:
            return self.annotator.annotate(scene, smoothed_detections, labels=labels, custom_color_lookup=custom_color_lookup)
        else:
            return self.annotator.annotate(scene, smoothed_detections, custom_color_lookup=custom_color_lookup)