import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RinkKeypoint:
    """Represents a keypoint on the rink."""
    id: int
    name: str
    position: Tuple[float, float]
    confidence: float


class RinkKeypointDetector:
    """
    Detects rink keypoints using YOLO pose estimation model.
    The model detects 56 keypoints representing various rink landmarks.
    """
    
    # Groups of keypoints by their likely function (based on observation)
    # These are estimates - actual mapping would need documentation
    KEYPOINT_GROUPS = {
        "left_zone": list(range(0, 20)),
        "center_zone": list(range(20, 36)),
        "right_zone": list(range(36, 56)),
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize the rink keypoint detector.
        
        Args:
            model_path: Path to the detection model
        """
        if model_path is None:
            model_path = "hockey/data/hockey-detection.pt"
        
        try:
            self.model = YOLO(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def detect_keypoints(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[RinkKeypoint]:
        """
        Detect rink keypoints from frame using pose estimation.
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for keypoints
            
        Returns:
            List of detected keypoints
        """
        if self.model is None:
            return []
        
        keypoints = []
        
        # Run pose estimation
        results = self.model(frame, verbose=False)
        
        if len(results) == 0:
            return keypoints
        
        result = results[0]
        
        # Check if keypoints were detected
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return keypoints
        
        # Get keypoint data
        if not hasattr(result.keypoints, 'data') or len(result.keypoints.data) == 0:
            return keypoints
        
        # Extract keypoints (first detection - should be the rink)
        kpts_data = result.keypoints.data[0]  # Shape: [56, 3] (x, y, confidence)
        
        # Process each keypoint
        for i, kpt in enumerate(kpts_data):
            x, y, conf = kpt[0].item(), kpt[1].item(), kpt[2].item()
            
            # Skip low confidence keypoints
            if conf < conf_threshold:
                continue
            
            # Determine keypoint group/zone
            zone = "unknown"
            for group_name, indices in self.KEYPOINT_GROUPS.items():
                if i in indices:
                    zone = group_name
                    break
            
            # Create keypoint
            keypoint = RinkKeypoint(
                id=i,
                name=f"{zone}_kpt_{i}",
                position=(x, y),
                confidence=conf
            )
            keypoints.append(keypoint)
        
        return keypoints
    
    def visualize_keypoints(
        self, 
        frame: np.ndarray, 
        keypoints: List[RinkKeypoint],
        radius: int = 8,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize detected keypoints on frame.
        
        Args:
            frame: Input frame
            keypoints: List of keypoints to visualize
            radius: Radius of keypoint circles
            show_labels: Whether to show keypoint labels
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Define colors for different keypoint zones
        colors = {
            "left_zone": (0, 255, 0),       # Green for left zone
            "center_zone": (255, 191, 0),   # Blue for center zone
            "right_zone": (71, 99, 255),    # Red for right zone
            "unknown": (255, 255, 255)      # White for unknown
        }
        
        for kp in keypoints:
            x, y = int(kp.position[0]), int(kp.position[1])
            
            # Determine color based on keypoint zone
            zone = kp.name.split('_')[0]  # Extract zone from name
            color = colors.get(zone, colors["unknown"])
            
            # Draw gradient circle effect
            for r in range(radius + 4, 0, -1):
                alpha = 1.0 - (r / (radius + 4))
                overlay_color = tuple(int(c * alpha) for c in color)
                cv2.circle(annotated, (x, y), r, overlay_color, -1)
            
            # Draw main keypoint
            cv2.circle(annotated, (x, y), radius, color, -1)
            cv2.circle(annotated, (x, y), radius, (255, 255, 255), 2)
            
            if show_labels:
                # Draw label
                label = f"{kp.id}:{kp.confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                (text_w, text_h), _ = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Background for text
                cv2.rectangle(
                    annotated,
                    (x - text_w // 2 - 2, y - radius - text_h - 4),
                    (x + text_w // 2 + 2, y - radius - 2),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated,
                    label,
                    (x - text_w // 2, y - radius - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        
        return annotated
    
    def get_rink_homography(self, keypoints: List[RinkKeypoint]) -> Optional[np.ndarray]:
        """
        Calculate homography matrix from detected keypoints.
        This would map rink coordinates to a standard rink template.
        
        Args:
            keypoints: Detected keypoints
            
        Returns:
            Homography matrix if enough keypoints are found
        """
        # This is a placeholder - would need actual rink dimensions
        # and correspondence points to calculate real homography
        
        # Filter keypoints by type and confidence
        center_points = [kp for kp in keypoints if "center_ice" in kp.name and kp.confidence > 0.7]
        faceoff_points = [kp for kp in keypoints if "faceoff" in kp.name and kp.confidence > 0.7]
        
        if len(center_points) >= 1 and len(faceoff_points) >= 4:
            # Would calculate homography here
            # For now, return None
            pass
        
        return None