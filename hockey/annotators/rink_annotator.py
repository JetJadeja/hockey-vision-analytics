import cv2
import supervision as sv
import numpy as np
from typing import List, Tuple, Optional, Dict

from configs.hockey import HockeyRinkConfiguration
from common.camera_view_detector import CameraViewDetector, ViewInfo
from common.view_mappers import AdaptiveMapper
from common.rink_keypoint_detector import RinkKeypoint

def draw_rink(
    config: HockeyRinkConfiguration,
    background_color: sv.Color = sv.Color.WHITE,
    line_color: sv.Color = sv.Color.BLUE,
    padding: int = 50,
    line_thickness: int = 4,
    scale: float = 2.0
) -> np.ndarray:
    """
    Draws a hockey rink with specified dimensions and colors.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    rink_image = np.full(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        background_color.as_bgr(),
        dtype=np.uint8
    )

    # Define colors
    blue_color = sv.Color.BLUE.as_bgr()
    red_color = sv.Color.RED.as_bgr()
    
    # Rink dimensions
    rink_left = padding
    rink_right = scaled_length + padding
    rink_top = padding
    rink_bottom = scaled_width + padding
    center_x = (rink_left + rink_right) // 2
    center_y = (rink_top + rink_bottom) // 2
    
    # Corner radius for rounded corners
    corner_radius = int(28 * scale)
    
    # Draw rink boundary with rounded corners
    # Main rectangle body
    cv2.rectangle(rink_image, 
                 (rink_left + corner_radius, rink_top), 
                 (rink_right - corner_radius, rink_bottom), 
                 blue_color, line_thickness)
    cv2.rectangle(rink_image, 
                 (rink_left, rink_top + corner_radius), 
                 (rink_right, rink_bottom - corner_radius), 
                 blue_color, line_thickness)
    
    # Draw rounded corners
    cv2.ellipse(rink_image, (rink_left + corner_radius, rink_top + corner_radius), 
               (corner_radius, corner_radius), 0, 180, 270, blue_color, line_thickness)
    cv2.ellipse(rink_image, (rink_right - corner_radius, rink_top + corner_radius), 
               (corner_radius, corner_radius), 0, 270, 360, blue_color, line_thickness)
    cv2.ellipse(rink_image, (rink_left + corner_radius, rink_bottom - corner_radius), 
               (corner_radius, corner_radius), 0, 90, 180, blue_color, line_thickness)
    cv2.ellipse(rink_image, (rink_right - corner_radius, rink_bottom - corner_radius), 
               (corner_radius, corner_radius), 0, 0, 90, blue_color, line_thickness)
    
    # Center red line
    cv2.line(rink_image, (center_x, rink_top), (center_x, rink_bottom), red_color, line_thickness)
    
    # Blue lines (divide rink into thirds)
    blue_line_1 = rink_left + scaled_length // 3
    blue_line_2 = rink_right - scaled_length // 3
    cv2.line(rink_image, (blue_line_1, rink_top), (blue_line_1, rink_bottom), blue_color, line_thickness)
    cv2.line(rink_image, (blue_line_2, rink_top), (blue_line_2, rink_bottom), blue_color, line_thickness)
    
    # Goal lines
    goal_line_distance = int(config.goal_line_from_end * scale)
    goal_line_1 = rink_left + goal_line_distance
    goal_line_2 = rink_right - goal_line_distance
    cv2.line(rink_image, (goal_line_1, rink_top), (goal_line_1, rink_bottom), red_color, line_thickness)
    cv2.line(rink_image, (goal_line_2, rink_top), (goal_line_2, rink_bottom), red_color, line_thickness)
    
    # Center faceoff circle (blue)
    center_circle_radius = int(config.faceoff_circle_radius * scale)
    cv2.circle(rink_image, (center_x, center_y), center_circle_radius, blue_color, line_thickness)
    cv2.circle(rink_image, (center_x, center_y), 3, blue_color, -1)  # Center dot
    
    # End zone faceoff circles (red)
    faceoff_dot_distance = int(config.faceoff_dot_to_goal_line * scale)
    faceoff_side_distance = int(config.faceoff_dot_from_side * scale)
    
    # Left zone circles
    left_faceoff_x = goal_line_1 + faceoff_dot_distance
    faceoff_y1 = rink_top + faceoff_side_distance
    faceoff_y2 = rink_bottom - faceoff_side_distance
    
    cv2.circle(rink_image, (left_faceoff_x, faceoff_y1), center_circle_radius, red_color, line_thickness)
    cv2.circle(rink_image, (left_faceoff_x, faceoff_y1), 3, red_color, -1)
    cv2.circle(rink_image, (left_faceoff_x, faceoff_y2), center_circle_radius, red_color, line_thickness)
    cv2.circle(rink_image, (left_faceoff_x, faceoff_y2), 3, red_color, -1)
    
    # Right zone circles
    right_faceoff_x = goal_line_2 - faceoff_dot_distance
    cv2.circle(rink_image, (right_faceoff_x, faceoff_y1), center_circle_radius, red_color, line_thickness)
    cv2.circle(rink_image, (right_faceoff_x, faceoff_y1), 3, red_color, -1)
    cv2.circle(rink_image, (right_faceoff_x, faceoff_y2), center_circle_radius, red_color, line_thickness)
    cv2.circle(rink_image, (right_faceoff_x, faceoff_y2), 3, red_color, -1)
    
    # Goal creases (semicircles in front of goals)
    goal_crease_radius = int(6 * scale)
    cv2.ellipse(rink_image, (goal_line_1, center_y), 
               (goal_crease_radius, goal_crease_radius), 0, 270, 90, blue_color, line_thickness)
    cv2.ellipse(rink_image, (goal_line_2, center_y), 
               (goal_crease_radius, goal_crease_radius), 0, 90, 270, blue_color, line_thickness)
    
    # Add neutral zone faceoff dots
    neutral_dot_distance = int(config.neutral_zone_dot_from_center * scale)
    neutral_dot_side_distance = int(config.neutral_zone_dot_from_side * scale)
    
    # Left neutral zone dots
    left_neutral_x = center_x - neutral_dot_distance
    neutral_y1 = rink_top + neutral_dot_side_distance
    neutral_y2 = rink_bottom - neutral_dot_side_distance
    cv2.circle(rink_image, (left_neutral_x, neutral_y1), 3, red_color, -1)
    cv2.circle(rink_image, (left_neutral_x, neutral_y2), 3, red_color, -1)
    
    # Right neutral zone dots
    right_neutral_x = center_x + neutral_dot_distance
    cv2.circle(rink_image, (right_neutral_x, neutral_y1), 3, red_color, -1)
    cv2.circle(rink_image, (right_neutral_x, neutral_y2), 3, red_color, -1)
    
    # Add faceoff circle hash marks
    hash_length = int(config.faceoff_hash_mark_length * scale)
    hash_distance = int(config.faceoff_hash_mark_distance * scale)
    
    def draw_faceoff_hash_marks(center_pos, color):
        cx, cy = center_pos
        circle_radius = center_circle_radius + hash_distance
        
        # Draw 4 hash marks around each circle (top, bottom, left, right)
        # Top hash marks
        cv2.line(rink_image, 
                (cx - hash_length//2, cy - circle_radius), 
                (cx + hash_length//2, cy - circle_radius), 
                color, line_thickness)
        # Bottom hash marks  
        cv2.line(rink_image, 
                (cx - hash_length//2, cy + circle_radius), 
                (cx + hash_length//2, cy + circle_radius), 
                color, line_thickness)
        # Left hash marks
        cv2.line(rink_image, 
                (cx - circle_radius, cy - hash_length//2), 
                (cx - circle_radius, cy + hash_length//2), 
                color, line_thickness)
        # Right hash marks
        cv2.line(rink_image, 
                (cx + circle_radius, cy - hash_length//2), 
                (cx + circle_radius, cy + hash_length//2), 
                color, line_thickness)
    
    # Add hash marks to all faceoff circles
    draw_faceoff_hash_marks((center_x, center_y), blue_color)  # Center circle
    draw_faceoff_hash_marks((left_faceoff_x, faceoff_y1), red_color)  # Left zone circles
    draw_faceoff_hash_marks((left_faceoff_x, faceoff_y2), red_color)
    draw_faceoff_hash_marks((right_faceoff_x, faceoff_y1), red_color)  # Right zone circles
    draw_faceoff_hash_marks((right_faceoff_x, faceoff_y2), red_color)

    return rink_image

class RinkMapVisualizer:
    """
    Visualizes player positions on a 2D rink map with camera-aware mapping.
    """
    
    def __init__(
        self, 
        config: HockeyRinkConfiguration,
        scale: float = 1.5,
        padding: int = 30,
        team_colors: Optional[Dict[int, tuple]] = None
    ):
        """
        Initialize the rink map visualizer.
        
        Args:
            config: Hockey rink configuration
            scale: Scale factor for the rink
            padding: Padding around the rink
            team_colors: Dictionary mapping team ID to BGR color tuple
        """
        self.config = config
        self.scale = scale
        self.padding = padding
        
        # Default team colors
        self.team_colors = team_colors or {
            0: (147, 20, 255),   # Deep pink for team 0
            1: (255, 191, 0),    # Blue for team 1
            2: (71, 99, 255)     # Red for goalies
        }
        
        # Create base rink image
        self.base_rink = draw_rink(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=2
        )
        
        # Calculate rink dimensions for coordinate mapping
        self.rink_width = int(config.width * scale)
        self.rink_length = int(config.length * scale)
        self.rink_left = padding
        self.rink_right = self.rink_length + padding
        self.rink_top = padding
        self.rink_bottom = self.rink_width + padding
        
        # Camera-aware components
        self.camera_detector = CameraViewDetector()
        self.adaptive_mapper = AdaptiveMapper()
        self.current_view_info = None
        
        # Store homography matrix if available
        self.homography = None
    
    def set_homography(self, homography: np.ndarray):
        """
        Set homography matrix for coordinate transformation.
        
        Args:
            homography: 3x3 homography matrix from video to rink coordinates
        """
        self.homography = homography
    
    def update_camera_view(self, keypoints: Optional[List[RinkKeypoint]], frame_shape: Tuple[int, int]):
        """
        Update camera view detection based on keypoints.
        
        Args:
            keypoints: Detected rink keypoints
            frame_shape: Shape of the video frame
        """
        self.current_view_info = self.camera_detector.classify_view(keypoints, frame_shape)
        self.adaptive_mapper.update_view(self.current_view_info)
    
    def transform_point(self, point: Tuple[float, float], video_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Transform a point from video coordinates to rink map coordinates.
        
        Args:
            point: (x, y) in video coordinates
            video_shape: (height, width) of the video frame
            
        Returns:
            (x, y) in rink map coordinates
        """
        if self.homography is not None:
            # Use homography if available
            pt = np.array([[point[0], point[1]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt.reshape(1, 1, 2), self.homography)
            map_x, map_y = transformed[0, 0]
            return int(map_x), int(map_y)
        else:
            # Use adaptive mapper based on camera view
            rink_dimensions = (self.rink_length, self.rink_width)
            return self.adaptive_mapper.transform_point(
                point, video_shape, rink_dimensions, self.padding
            )
    
    def draw_players(
        self,
        frame_shape: Tuple[int, int],
        player_positions: List[Tuple[float, float]],
        team_assignments: np.ndarray,
        player_radius: int = 8
    ) -> np.ndarray:
        """
        Draw players on the rink map.
        
        Args:
            frame_shape: Shape of the video frame (height, width)
            player_positions: List of (x, y) positions in video coordinates
            team_assignments: Array of team IDs for each player
            player_radius: Radius of player dots
            
        Returns:
            Rink map with players drawn
        """
        # Create a copy of the base rink
        rink_map = self.base_rink.copy()
        
        # Draw each player
        for i, (pos, team_id) in enumerate(zip(player_positions, team_assignments)):
            # Transform position to rink coordinates
            map_x, map_y = self.transform_point(pos, frame_shape)
            
            # Get team color
            color = self.team_colors.get(team_id, (255, 255, 255))
            
            # Draw player dot with outline
            cv2.circle(rink_map, (map_x, map_y), player_radius, color, -1)
            cv2.circle(rink_map, (map_x, map_y), player_radius, (0, 0, 0), 2)
            
            # Optional: Add player number if available
            # cv2.putText(rink_map, str(i), (map_x-5, map_y+5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return rink_map
    
    def create_combined_view(
        self,
        video_frame: np.ndarray,
        player_positions: List[Tuple[float, float]],
        team_assignments: np.ndarray,
        gap: int = 20
    ) -> np.ndarray:
        """
        Create a combined view with video on top and rink map below.
        
        Args:
            video_frame: The annotated video frame
            player_positions: List of player positions
            team_assignments: Array of team IDs
            gap: Gap between video and map
            
        Returns:
            Combined frame
        """
        # Get rink map with players
        rink_map = self.draw_players(
            frame_shape=(video_frame.shape[0], video_frame.shape[1]),
            player_positions=player_positions,
            team_assignments=team_assignments
        )
        
        # Resize rink map to match video width
        video_height, video_width = video_frame.shape[:2]
        map_height, map_width = rink_map.shape[:2]
        
        # Calculate scaling to fit width
        scale_factor = video_width / map_width
        new_map_height = int(map_height * scale_factor)
        new_map_width = video_width
        
        # Resize rink map
        resized_map = cv2.resize(rink_map, (new_map_width, new_map_height))
        
        # Create combined frame
        total_height = video_height + gap + new_map_height
        combined = np.zeros((total_height, video_width, 3), dtype=np.uint8)
        
        # Place video on top
        combined[:video_height, :] = video_frame
        
        # Add separator line
        combined[video_height:video_height+gap, :] = (128, 128, 128)
        
        # Place rink map below
        combined[video_height+gap:, :] = resized_map
        
        return combined


if __name__ == "__main__":
    # Create a standard hockey rink configuration
    config = HockeyRinkConfiguration()  # Use default values

    # Draw the rink with better scaling
    rink_image = draw_rink(config, scale=3.0, padding=30)

    # Display the image using OpenCV
    cv2.imshow("Hockey Rink", rink_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()