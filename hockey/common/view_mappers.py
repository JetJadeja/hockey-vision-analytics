import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from .camera_view_detector import CameraView, ViewInfo


class ViewMapper(ABC):
    """Abstract base class for view-specific coordinate mappers."""
    
    @abstractmethod
    def transform_point(
        self, 
        point: Tuple[float, float], 
        frame_shape: Tuple[int, int],
        rink_dimensions: Tuple[float, float],
        padding: int
    ) -> Tuple[int, int]:
        """Transform a point from video coordinates to rink map coordinates."""
        pass


class BroadcastWideMapper(ViewMapper):
    """Mapper for standard broadcast wide view (full rink side view)."""
    
    def transform_point(
        self, 
        point: Tuple[float, float], 
        frame_shape: Tuple[int, int],
        rink_dimensions: Tuple[float, float],
        padding: int
    ) -> Tuple[int, int]:
        """Transform point for broadcast wide view."""
        video_height, video_width = frame_shape
        rink_length, rink_width = rink_dimensions
        
        # Normalize coordinates
        norm_x = point[0] / video_width
        norm_y = point[1] / video_height
        
        # Apply slight perspective correction (farther players are higher in frame)
        # Adjust y based on position in frame
        perspective_factor = 1.0 + (norm_y - 0.5) * 0.1
        
        # Map to rink coordinates (rink is horizontal)
        map_x = padding + norm_x * rink_length
        map_y = padding + norm_y * rink_width * perspective_factor
        
        return int(map_x), int(map_y)


class BroadcastZoneMapper(ViewMapper):
    """Mapper for broadcast zone view (partial rink from stands)."""
    
    def __init__(self, visible_zones: List[str]):
        self.visible_zones = visible_zones
        
    def transform_point(
        self, 
        point: Tuple[float, float], 
        frame_shape: Tuple[int, int],
        rink_dimensions: Tuple[float, float],
        padding: int
    ) -> Tuple[int, int]:
        """Transform point for broadcast zone view."""
        video_height, video_width = frame_shape
        rink_length, rink_width = rink_dimensions
        
        # Determine which portion of rink is visible
        if "left" in self.visible_zones and "right" not in self.visible_zones:
            # Left offensive zone
            x_offset = 0
            x_scale = rink_length / 3
        elif "right" in self.visible_zones and "left" not in self.visible_zones:
            # Right offensive zone
            x_offset = 2 * rink_length / 3
            x_scale = rink_length / 3
        else:
            # Multiple zones visible
            x_offset = 0
            x_scale = rink_length
        
        # Normalize coordinates
        norm_x = point[0] / video_width
        norm_y = point[1] / video_height
        
        # Apply perspective correction
        perspective_factor = 1.0 + (norm_y - 0.5) * 0.15
        
        # Map to rink coordinates
        map_x = padding + x_offset + norm_x * x_scale
        map_y = padding + norm_y * rink_width * perspective_factor
        
        return int(map_x), int(map_y)


class BehindGoalMapper(ViewMapper):
    """Mapper for behind-goal camera view."""
    
    def __init__(self, zone: str):
        self.zone = zone  # "left" or "right"
        
    def transform_point(
        self, 
        point: Tuple[float, float], 
        frame_shape: Tuple[int, int],
        rink_dimensions: Tuple[float, float],
        padding: int
    ) -> Tuple[int, int]:
        """Transform point for behind-goal view."""
        video_height, video_width = frame_shape
        rink_length, rink_width = rink_dimensions
        
        # Normalize coordinates
        norm_x = point[0] / video_width
        norm_y = point[1] / video_height
        
        # For behind-goal view, the rink appears vertical
        # X in video maps to Y on rink (side to side)
        # Y in video maps to X on rink (distance from goal)
        
        # Strong perspective correction (closer = lower in frame)
        depth_factor = (1.0 - norm_y) ** 1.5  # Exponential perspective
        
        if self.zone == "left":
            # Behind left goal
            map_x = padding + (1.0 - depth_factor) * (rink_length / 3)
            map_y = padding + norm_x * rink_width
        else:
            # Behind right goal
            map_x = padding + rink_length - (1.0 - depth_factor) * (rink_length / 3)
            map_y = padding + (1.0 - norm_x) * rink_width
        
        return int(map_x), int(map_y)


class AdaptiveMapper:
    """Adaptive mapper that selects appropriate mapper based on view info."""
    
    def __init__(self):
        self.current_mapper = None
        self.view_info = None
        
    def update_view(self, view_info: ViewInfo):
        """Update the mapper based on new view information."""
        self.view_info = view_info
        
        # Select appropriate mapper
        if view_info.view_type == CameraView.BROADCAST_WIDE:
            self.current_mapper = BroadcastWideMapper()
        elif view_info.view_type == CameraView.BROADCAST_ZONE:
            self.current_mapper = BroadcastZoneMapper(view_info.visible_zones)
        elif view_info.view_type == CameraView.BEHIND_GOAL:
            # Determine which goal based on visible zone
            zone = view_info.visible_zones[0] if view_info.visible_zones else "left"
            self.current_mapper = BehindGoalMapper(zone)
        else:
            # Default to broadcast wide
            self.current_mapper = BroadcastWideMapper()
    
    def transform_point(
        self, 
        point: Tuple[float, float], 
        frame_shape: Tuple[int, int],
        rink_dimensions: Tuple[float, float],
        padding: int
    ) -> Tuple[int, int]:
        """Transform point using current mapper."""
        if self.current_mapper is None:
            # Default mapper
            self.current_mapper = BroadcastWideMapper()
        
        return self.current_mapper.transform_point(
            point, frame_shape, rink_dimensions, padding
        )