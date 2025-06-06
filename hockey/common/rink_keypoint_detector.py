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
        Calculate homography matrix from detected keypoints using pre-mapped positions.
        Maps video coordinates to rink coordinates (in pixels).
        
        Args:
            keypoints: Detected keypoints
            
        Returns:
            Homography matrix if enough keypoints are found, None otherwise
        """
        import json
        import os
        
        # Load keypoint mappings
        keypoints_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'keypoints.json')
        if not os.path.exists(keypoints_file):
            print(f"Warning: keypoints.json not found at {keypoints_file}")
            return None
        
        try:
            with open(keypoints_file, 'r') as f:
                keypoint_data = json.load(f)
        except Exception as e:
            print(f"Error loading keypoints.json: {e}")
            return None
        
        # Extract mapped keypoints
        mapped_keypoints = keypoint_data.get('keypoints', {})
        
        # Collect point correspondences
        src_points = []  # Video coordinates
        dst_points = []  # Rink coordinates (in pixels)
        
        # Scale and padding must match draw_rink exactly
        scale = 3.0
        padding = 50
        
        # Build correspondences from detected keypoints
        for kp in keypoints:
            kp_id_str = str(kp.id)
            
            # Check if this keypoint has a mapping and sufficient confidence
            if kp_id_str in mapped_keypoints and kp.confidence > 0.5:
                mapping = mapped_keypoints[kp_id_str]
                if mapping.get('rink_position'):
                    # Get video coordinates
                    src_points.append(kp.position)
                    
                    # Convert feet to pixels (matching draw_rink)
                    rink_x_ft, rink_y_ft = mapping['rink_position']
                    dst_x_px = rink_x_ft * scale + padding
                    dst_y_px = rink_y_ft * scale + padding
                    dst_points.append((dst_x_px, dst_y_px))
        
        # Need at least 4 points for homography
        if len(src_points) < 4:
            return None
        
        # Convert to numpy arrays
        src_pts = np.array(src_points, dtype=np.float32)
        dst_pts = np.array(dst_points, dtype=np.float32)
        
        # Calculate homography using RANSAC for robustness
        try:
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if homography is not None:
                # Count inliers
                inliers = np.sum(mask) if mask is not None else 0
                total = len(src_points)
                
                # Require at least 4 inliers for valid homography
                if inliers >= 4:
                    print(f"Homography calculated: {inliers}/{total} inliers")
                    return homography
                else:
                    print(f"Homography rejected: only {inliers}/{total} inliers")
                    return None
        except Exception as e:
            print(f"Error calculating homography: {e}")
            return None
        
        return None