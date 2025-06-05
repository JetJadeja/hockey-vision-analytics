import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .rink_keypoint_detector import RinkKeypoint


class CameraView(Enum):
    """Different camera view types in hockey broadcasts."""
    BROADCAST_WIDE = "broadcast_wide"      # Full rink side view
    BROADCAST_ZONE = "broadcast_zone"      # Partial rink, angled from stands
    BEHIND_GOAL = "behind_goal"           # View from behind one goal
    OVERHEAD = "overhead"                 # Top-down or near top-down
    UNKNOWN = "unknown"                   # Cannot determine view


@dataclass
class ViewInfo:
    """Information about the detected camera view."""
    view_type: CameraView
    confidence: float
    visible_zones: List[str]  # ["left", "neutral", "right"]
    rotation_angle: float     # Estimated rotation from standard view
    visible_percentage: float # Percentage of rink visible


class CameraViewDetector:
    """Detects the camera view type based on rink keypoints and frame analysis."""
    
    def __init__(self):
        # Keypoint indices for different rink elements (based on 56-keypoint model)
        self.left_zone_indices = set(range(0, 20))
        self.center_zone_indices = set(range(20, 36))
        self.right_zone_indices = set(range(36, 56))
        
    def classify_view(
        self, 
        keypoints: Optional[List[RinkKeypoint]], 
        frame_shape: Tuple[int, int]
    ) -> ViewInfo:
        """
        Classify the camera view based on keypoints and frame shape.
        
        Args:
            keypoints: Detected rink keypoints
            frame_shape: (height, width) of the frame
            
        Returns:
            ViewInfo with detected camera view information
        """
        if not keypoints or len(keypoints) < 3:
            return ViewInfo(
                view_type=CameraView.UNKNOWN,
                confidence=0.0,
                visible_zones=[],
                rotation_angle=0.0,
                visible_percentage=0.0
            )
        
        # Analyze keypoint distribution
        zones_visible = self._analyze_visible_zones(keypoints)
        keypoint_spread = self._calculate_keypoint_spread(keypoints, frame_shape)
        aspect_ratio = frame_shape[1] / frame_shape[0]  # width/height
        
        # Determine view type based on analysis
        if len(zones_visible) == 3 and keypoint_spread['horizontal'] > 0.7:
            # All zones visible with wide horizontal spread
            return ViewInfo(
                view_type=CameraView.BROADCAST_WIDE,
                confidence=0.9,
                visible_zones=zones_visible,
                rotation_angle=0.0,
                visible_percentage=0.9
            )
        
        elif len(zones_visible) == 1:
            # Only one zone visible
            if keypoint_spread['vertical'] > keypoint_spread['horizontal']:
                # Vertical orientation suggests behind-goal view
                return ViewInfo(
                    view_type=CameraView.BEHIND_GOAL,
                    confidence=0.8,
                    visible_zones=zones_visible,
                    rotation_angle=90.0,
                    visible_percentage=0.3
                )
            else:
                # Horizontal orientation suggests broadcast zone view
                return ViewInfo(
                    view_type=CameraView.BROADCAST_ZONE,
                    confidence=0.8,
                    visible_zones=zones_visible,
                    rotation_angle=0.0,
                    visible_percentage=0.4
                )
        
        elif len(zones_visible) == 2:
            # Two zones visible - likely broadcast zone
            return ViewInfo(
                view_type=CameraView.BROADCAST_ZONE,
                confidence=0.7,
                visible_zones=zones_visible,
                rotation_angle=0.0,
                visible_percentage=0.6
            )
        
        # Default to unknown
        return ViewInfo(
            view_type=CameraView.UNKNOWN,
            confidence=0.3,
            visible_zones=zones_visible,
            rotation_angle=0.0,
            visible_percentage=0.5
        )
    
    def _analyze_visible_zones(self, keypoints: List[RinkKeypoint]) -> List[str]:
        """Determine which zones are visible based on keypoint indices."""
        visible_zones = []
        
        # Count keypoints in each zone
        left_count = sum(1 for kp in keypoints if kp.id in self.left_zone_indices)
        center_count = sum(1 for kp in keypoints if kp.id in self.center_zone_indices)
        right_count = sum(1 for kp in keypoints if kp.id in self.right_zone_indices)
        
        # Threshold for considering a zone visible
        min_keypoints = 2
        
        if left_count >= min_keypoints:
            visible_zones.append("left")
        if center_count >= min_keypoints:
            visible_zones.append("neutral")
        if right_count >= min_keypoints:
            visible_zones.append("right")
        
        return visible_zones
    
    def _calculate_keypoint_spread(
        self, 
        keypoints: List[RinkKeypoint], 
        frame_shape: Tuple[int, int]
    ) -> dict:
        """Calculate the spread of keypoints in horizontal and vertical directions."""
        if not keypoints:
            return {'horizontal': 0.0, 'vertical': 0.0}
        
        # Extract positions
        positions = np.array([(kp.position[0], kp.position[1]) for kp in keypoints])
        
        # Calculate normalized spread
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        horizontal_spread = (x_max - x_min) / frame_shape[1]
        vertical_spread = (y_max - y_min) / frame_shape[0]
        
        return {
            'horizontal': horizontal_spread,
            'vertical': vertical_spread
        }