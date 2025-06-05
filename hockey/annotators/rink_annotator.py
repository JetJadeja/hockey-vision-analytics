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
    Draws a professional NHL-standard hockey rink with accurate dimensions and markings.
    
    Official NHL Specifications:
    - Rink: 200' x 85' (61m x 26m)
    - Corner radius: 28' (8.5m)
    - Goal line: 11' (3.4m) from end boards
    - Blue lines: 60' (18.3m) from goal lines, 12" (30cm) wide
    - Center line: 12" (30cm) wide red line
    - Face-off circles: 30' (9.1m) diameter
    - Goal crease: 6' (1.8m) radius semicircle
    """
    # Use accurate NHL dimensions (200' x 85')
    scaled_length = int(200 * scale)  # Length (end to end)
    scaled_width = int(85 * scale)    # Width (side to side)
    
    # Create canvas with padding
    canvas_width = scaled_length + 2 * padding
    canvas_height = scaled_width + 2 * padding
    rink_image = np.full((canvas_height, canvas_width, 3), background_color.as_bgr(), dtype=np.uint8)
    
    # Define professional hockey colors
    ice_white = (248, 248, 250)          # Slightly off-white ice color
    board_color = (45, 45, 45)           # Dark gray boards
    red_line_color = (0, 0, 200)         # NHL red
    blue_line_color = (200, 0, 0)        # NHL blue
    goal_crease_color = (150, 150, 255)  # Light blue for goal crease
    
    # Calculate rink boundaries
    rink_left = padding
    rink_right = padding + scaled_length
    rink_top = padding
    rink_bottom = padding + scaled_width
    center_x = padding + scaled_length // 2
    center_y = padding + scaled_width // 2
    
    # Corner radius (28 feet in NHL)
    corner_radius = int(28 * scale)
    
    # Fill ice surface with ice white
    cv2.rectangle(rink_image, (rink_left, rink_top), (rink_right, rink_bottom), ice_white, -1)
    
    # Draw boards (perimeter) with proper corner radius
    board_thickness = max(3, int(line_thickness * 1.5))
    
    # Main rectangles for boards
    cv2.rectangle(rink_image, 
                 (rink_left + corner_radius, rink_top), 
                 (rink_right - corner_radius, rink_bottom), 
                 board_color, board_thickness)
    cv2.rectangle(rink_image, 
                 (rink_left, rink_top + corner_radius), 
                 (rink_right, rink_bottom - corner_radius), 
                 board_color, board_thickness)
    
    # Draw rounded corners
    cv2.ellipse(rink_image, (rink_left + corner_radius, rink_top + corner_radius), 
               (corner_radius, corner_radius), 0, 180, 270, board_color, board_thickness)
    cv2.ellipse(rink_image, (rink_right - corner_radius, rink_top + corner_radius), 
               (corner_radius, corner_radius), 0, 270, 360, board_color, board_thickness)
    cv2.ellipse(rink_image, (rink_left + corner_radius, rink_bottom - corner_radius), 
               (corner_radius, corner_radius), 0, 90, 180, board_color, board_thickness)
    cv2.ellipse(rink_image, (rink_right - corner_radius, rink_bottom - corner_radius), 
               (corner_radius, corner_radius), 0, 0, 90, board_color, board_thickness)
    
    # Goal lines (11 feet from end boards)
    goal_line_distance = int(11 * scale)
    left_goal_line = rink_left + goal_line_distance
    right_goal_line = rink_right - goal_line_distance
    goal_line_thickness = max(2, int(line_thickness * 0.75))
    
    cv2.line(rink_image, (left_goal_line, rink_top), (left_goal_line, rink_bottom), 
             red_line_color, goal_line_thickness)
    cv2.line(rink_image, (right_goal_line, rink_top), (right_goal_line, rink_bottom), 
             red_line_color, goal_line_thickness)
    
    # Blue lines (60 feet from goal lines, 12" wide)
    blue_line_distance = int(60 * scale)
    left_blue_line = left_goal_line + blue_line_distance
    right_blue_line = right_goal_line - blue_line_distance
    blue_line_thickness = max(6, int(line_thickness * 2))  # 12" wide
    
    cv2.line(rink_image, (left_blue_line, rink_top), (left_blue_line, rink_bottom), 
             blue_line_color, blue_line_thickness)
    cv2.line(rink_image, (right_blue_line, rink_top), (right_blue_line, rink_bottom), 
             blue_line_color, blue_line_thickness)
    
    # Center red line (12" wide)
    center_line_thickness = max(6, int(line_thickness * 2))  # 12" wide
    cv2.line(rink_image, (center_x, rink_top), (center_x, rink_bottom), 
             red_line_color, center_line_thickness)
    
    # Face-off circles (30 feet diameter)
    faceoff_radius = int(15 * scale)  # 15 feet radius = 30 feet diameter
    circle_thickness = max(2, int(line_thickness * 0.5))
    
    # Center ice face-off circle (blue)
    cv2.circle(rink_image, (center_x, center_y), faceoff_radius, blue_line_color, circle_thickness)
    
    # Center ice face-off spot
    spot_radius = max(6, int(scale * 1.5))
    cv2.circle(rink_image, (center_x, center_y), spot_radius, blue_line_color, -1)
    
    # End zone face-off circles and spots
    # Positioning: 20 feet from goal line, 22 feet from boards
    faceoff_from_goal = int(20 * scale)
    faceoff_from_side = int(22 * scale)
    
    left_faceoff_x = left_goal_line + faceoff_from_goal
    right_faceoff_x = right_goal_line - faceoff_from_goal
    faceoff_y1 = rink_top + faceoff_from_side
    faceoff_y2 = rink_bottom - faceoff_from_side
    
    # Draw end zone circles (red)
    for fx, fy in [(left_faceoff_x, faceoff_y1), (left_faceoff_x, faceoff_y2),
                   (right_faceoff_x, faceoff_y1), (right_faceoff_x, faceoff_y2)]:
        cv2.circle(rink_image, (fx, fy), faceoff_radius, red_line_color, circle_thickness)
        cv2.circle(rink_image, (fx, fy), spot_radius, red_line_color, -1)
    
    # Goal creases (6 feet radius semicircle)
    crease_radius = int(6 * scale)
    crease_thickness = max(2, int(line_thickness * 0.6))
    
    # Left goal crease
    cv2.ellipse(rink_image, (left_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 270, 90, 
               goal_crease_color, crease_thickness)
    # Fill crease area lightly
    cv2.ellipse(rink_image, (left_goal_line, center_y), 
               (crease_radius-1, crease_radius-1), 0, 270, 90, 
               goal_crease_color, -1)
    cv2.ellipse(rink_image, (left_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 270, 90, 
               goal_crease_color, crease_thickness)
    
    # Right goal crease  
    cv2.ellipse(rink_image, (right_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 90, 270, 
               goal_crease_color, crease_thickness)
    cv2.ellipse(rink_image, (right_goal_line, center_y), 
               (crease_radius-1, crease_radius-1), 0, 90, 270, 
               goal_crease_color, -1)
    cv2.ellipse(rink_image, (right_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 90, 270, 
               goal_crease_color, crease_thickness)
    
    # Neutral zone face-off spots
    neutral_distance = int(5 * scale)  # 5 feet from center line
    neutral_from_side = int(22 * scale)  # 22 feet from boards
    
    for nx in [center_x - neutral_distance, center_x + neutral_distance]:
        for ny in [rink_top + neutral_from_side, rink_bottom - neutral_from_side]:
            cv2.circle(rink_image, (nx, ny), spot_radius, red_line_color, -1)
    
    # Face-off circle hash marks
    hash_length = int(2 * scale)  # 2 feet long hash marks
    hash_distance = int(2 * scale)  # 2 feet outside circle
    hash_thickness = max(2, int(line_thickness * 0.5))
    
    def draw_hash_marks(center_pos, color, is_center=False):
        cx, cy = center_pos
        outer_radius = faceoff_radius + hash_distance
        
        # Only draw hash marks for end zone circles
        if not is_center:
            # Top and bottom hash marks
            cv2.line(rink_image, 
                    (cx - hash_length//2, cy - outer_radius), 
                    (cx + hash_length//2, cy - outer_radius), 
                    color, hash_thickness)
            cv2.line(rink_image, 
                    (cx - hash_length//2, cy + outer_radius), 
                    (cx + hash_length//2, cy + outer_radius), 
                    color, hash_thickness)
            # Left and right hash marks
            cv2.line(rink_image, 
                    (cx - outer_radius, cy - hash_length//2), 
                    (cx - outer_radius, cy + hash_length//2), 
                    color, hash_thickness)
            cv2.line(rink_image, 
                    (cx + outer_radius, cy - hash_length//2), 
                    (cx + outer_radius, cy + hash_length//2), 
                    color, hash_thickness)
    
    # Add hash marks to end zone circles
    for fx, fy in [(left_faceoff_x, faceoff_y1), (left_faceoff_x, faceoff_y2),
                   (right_faceoff_x, faceoff_y1), (right_faceoff_x, faceoff_y2)]:
        draw_hash_marks((fx, fy), red_line_color, False)
    
    # Goals (represented as small rectangles on goal line)
    goal_width = int(6 * scale)  # 6 feet wide
    goal_depth = int(2 * scale)  # 2 feet deep
    goal_thickness = max(2, int(line_thickness * 0.8))
    
    # Left goal
    goal_top = center_y - goal_width // 2
    goal_bottom = center_y + goal_width // 2
    cv2.rectangle(rink_image, 
                 (left_goal_line - goal_depth, goal_top),
                 (left_goal_line, goal_bottom),
                 red_line_color, goal_thickness)
    
    # Right goal
    cv2.rectangle(rink_image, 
                 (right_goal_line, goal_top),
                 (right_goal_line + goal_depth, goal_bottom),
                 red_line_color, goal_thickness)
    
    # Add referee crease (small semicircle behind center line)
    ref_crease_radius = int(3 * scale)
    cv2.ellipse(rink_image, (center_x, rink_bottom), 
               (ref_crease_radius, ref_crease_radius), 0, 0, 180, 
               red_line_color, max(1, circle_thickness // 2))
    
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