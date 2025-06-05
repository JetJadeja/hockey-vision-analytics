import cv2
import numpy as np
import supervision as sv
from typing import Dict, Optional


class SegmentationAnnotator:
    """
    Annotator that visualizes player segmentation masks on the frame.
    """
    
    def __init__(self, alpha: float = 0.5, team_colors: Optional[Dict[int, tuple]] = None):
        """
        Initialize the segmentation annotator.
        
        Args:
            alpha: Transparency level for overlay (0-1)
            team_colors: Dictionary mapping team ID to BGR color tuple
        """
        self.alpha = alpha
        self.team_colors = team_colors or {
            0: (255, 255, 255),  # White for away team
            1: (255, 20, 147),   # Deep pink for home team
            2: (71, 99, 255)     # Blue for goalies
        }
    
    def annotate(
        self, 
        scene: np.ndarray, 
        detections: sv.Detections,
        masks: Dict[int, np.ndarray],
        team_assignments: np.ndarray
    ) -> np.ndarray:
        """
        Annotate frame with segmentation masks.
        
        Args:
            scene: The frame to annotate
            detections: Detection results with tracker IDs
            masks: Dictionary mapping tracker ID to segmentation mask
            team_assignments: Array of team IDs for each detection
            
        Returns:
            Annotated frame
        """
        annotated_frame = scene.copy()
        overlay = scene.copy()
        
        # Process each detection
        for i, (xyxy, tracker_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            if tracker_id is None or int(tracker_id) not in masks:
                continue
                
            tracker_id = int(tracker_id)
            mask = masks[tracker_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = xyxy.astype(int)
            
            # Ensure mask matches crop dimensions
            crop_height = y2 - y1
            crop_width = x2 - x1
            
            if mask.shape[:2] != (crop_height, crop_width):
                # Resize mask to match crop dimensions
                mask = cv2.resize(mask.astype(np.uint8), (crop_width, crop_height))
                mask = mask.astype(bool)
            
            # Get team color
            team_id = team_assignments[i] if i < len(team_assignments) else 0
            color = self.team_colors.get(team_id, (255, 255, 255))
            
            # Create colored mask
            mask_colored = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
            mask_colored[mask] = color
            
            # Apply mask to overlay within bounding box
            overlay[y1:y2, x1:x2][mask] = mask_colored[mask]
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.alpha, annotated_frame, 1 - self.alpha, 0, annotated_frame)
        
        return annotated_frame