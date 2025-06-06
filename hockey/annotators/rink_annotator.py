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
    Draws a professional NHL-standard hockey rink with proper proportions and realistic appearance.
    """
    # NHL dimensions: 200' x 85' - maintain proper aspect ratio
    rink_length = 200  # feet
    rink_width = 85    # feet
    
    # Scale for proper proportions
    scaled_length = int(rink_length * scale)
    scaled_width = int(rink_width * scale)
    
    # Create canvas
    canvas_width = scaled_length + 2 * padding
    canvas_height = scaled_width + 2 * padding
    
    # Professional ice color - clean white with very subtle blue tint
    ice_color = (255, 255, 255)
    rink_image = np.full((canvas_height, canvas_width, 3), ice_color, dtype=np.uint8)
    
    # Professional hockey colors
    board_color = (50, 50, 50)        # Dark gray boards
    red_line_color = (0, 0, 255)      # Bright red (BGR format)
    blue_line_color = (255, 0, 0)     # Bright blue (BGR format)
    goal_crease_color = (255, 200, 200)  # Light blue
    
    # Rink boundaries
    rink_left = padding
    rink_right = padding + scaled_length
    rink_top = padding
    rink_bottom = padding + scaled_width
    center_x = padding + scaled_length // 2
    center_y = padding + scaled_width // 2
    
    # Dimensions in scale
    corner_radius = int(28 * scale)
    goal_line_from_end = int(11 * scale)
    blue_line_from_goal = int(64 * scale)  # 75' from end - 11' goal line = 64'
    
    # Line thicknesses - much thinner and more realistic
    board_thickness = max(3, int(scale * 1.5))
    center_line_thickness = max(2, int(scale * 0.8))  # Center red line
    blue_line_thickness = max(2, int(scale * 0.8))    # Blue lines
    goal_line_thickness = max(1, int(scale * 0.5))    # Goal lines
    circle_thickness = max(1, int(scale * 0.4))       # Face-off circles
    
    # Draw boards (rink perimeter)
    # Main rectangles
    cv2.rectangle(rink_image, 
                 (rink_left + corner_radius, rink_top), 
                 (rink_right - corner_radius, rink_bottom), 
                 board_color, board_thickness, cv2.LINE_AA)
    cv2.rectangle(rink_image, 
                 (rink_left, rink_top + corner_radius), 
                 (rink_right, rink_bottom - corner_radius), 
                 board_color, board_thickness, cv2.LINE_AA)
    
    # Rounded corners
    cv2.ellipse(rink_image, (rink_left + corner_radius, rink_top + corner_radius), 
               (corner_radius, corner_radius), 0, 180, 270, board_color, board_thickness, cv2.LINE_AA)
    cv2.ellipse(rink_image, (rink_right - corner_radius, rink_top + corner_radius), 
               (corner_radius, corner_radius), 0, 270, 360, board_color, board_thickness, cv2.LINE_AA)
    cv2.ellipse(rink_image, (rink_left + corner_radius, rink_bottom - corner_radius), 
               (corner_radius, corner_radius), 0, 90, 180, board_color, board_thickness, cv2.LINE_AA)
    cv2.ellipse(rink_image, (rink_right - corner_radius, rink_bottom - corner_radius), 
               (corner_radius, corner_radius), 0, 0, 90, board_color, board_thickness, cv2.LINE_AA)
    
    # Goal lines (11' from ends)
    left_goal_line = rink_left + goal_line_from_end
    right_goal_line = rink_right - goal_line_from_end
    
    cv2.line(rink_image, (left_goal_line, rink_top + board_thickness), 
             (left_goal_line, rink_bottom - board_thickness), 
             red_line_color, goal_line_thickness, cv2.LINE_AA)
    cv2.line(rink_image, (right_goal_line, rink_top + board_thickness), 
             (right_goal_line, rink_bottom - board_thickness), 
             red_line_color, goal_line_thickness, cv2.LINE_AA)
    
    # Blue lines (64' from goal lines)
    left_blue_line = left_goal_line + blue_line_from_goal
    right_blue_line = right_goal_line - blue_line_from_goal
    
    cv2.line(rink_image, (left_blue_line, rink_top + board_thickness), 
             (left_blue_line, rink_bottom - board_thickness), 
             blue_line_color, blue_line_thickness, cv2.LINE_AA)
    cv2.line(rink_image, (right_blue_line, rink_top + board_thickness), 
             (right_blue_line, rink_bottom - board_thickness), 
             blue_line_color, blue_line_thickness, cv2.LINE_AA)
    
    # Center red line
    cv2.line(rink_image, (center_x, rink_top + board_thickness), 
             (center_x, rink_bottom - board_thickness), 
             red_line_color, center_line_thickness, cv2.LINE_AA)
    
    # Face-off circles - proper size (15' radius = 30' diameter)
    faceoff_radius = int(15 * scale)
    
    # Center ice face-off circle
    cv2.circle(rink_image, (center_x, center_y), faceoff_radius, blue_line_color, circle_thickness, cv2.LINE_AA)
    
    # Center face-off spot
    spot_radius = max(2, int(scale * 0.5))
    cv2.circle(rink_image, (center_x, center_y), spot_radius, blue_line_color, -1, cv2.LINE_AA)
    
    # End zone face-off circles
    faceoff_from_goal = int(20 * scale)    # 20' from goal line
    faceoff_from_side = int(22 * scale)    # 22' from boards
    
    left_faceoff_x = left_goal_line + faceoff_from_goal
    right_faceoff_x = right_goal_line - faceoff_from_goal
    faceoff_y1 = rink_top + faceoff_from_side
    faceoff_y2 = rink_bottom - faceoff_from_side
    
    # Draw end zone circles and spots
    for fx, fy in [(left_faceoff_x, faceoff_y1), (left_faceoff_x, faceoff_y2),
                   (right_faceoff_x, faceoff_y1), (right_faceoff_x, faceoff_y2)]:
        cv2.circle(rink_image, (fx, fy), faceoff_radius, red_line_color, circle_thickness, cv2.LINE_AA)
        cv2.circle(rink_image, (fx, fy), spot_radius, red_line_color, -1, cv2.LINE_AA)
    
    # Goal creases (6' radius) - proper light blue color
    crease_radius = int(6 * scale)
    crease_thickness = max(1, int(scale * 0.4))
    
    # Left goal crease
    cv2.ellipse(rink_image, (left_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 270, 90, 
               goal_crease_color, -1, cv2.LINE_AA)
    cv2.ellipse(rink_image, (left_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 270, 90, 
               blue_line_color, crease_thickness, cv2.LINE_AA)
    
    # Right goal crease
    cv2.ellipse(rink_image, (right_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 90, 270, 
               goal_crease_color, -1, cv2.LINE_AA)
    cv2.ellipse(rink_image, (right_goal_line, center_y), 
               (crease_radius, crease_radius), 0, 90, 270, 
               blue_line_color, crease_thickness, cv2.LINE_AA)
    
    # Neutral zone face-off spots (smaller)
    neutral_distance = int(5 * scale)
    neutral_from_side = int(22 * scale)
    neutral_spot_radius = max(1, int(scale * 0.3))
    
    for nx in [center_x - neutral_distance, center_x + neutral_distance]:
        for ny in [rink_top + neutral_from_side, rink_bottom - neutral_from_side]:
            cv2.circle(rink_image, (nx, ny), neutral_spot_radius, red_line_color, -1, cv2.LINE_AA)
    
    # Face-off circle hash marks - small and visible
    hash_length = int(2 * scale)
    hash_distance = int(2 * scale)
    hash_thickness = max(1, int(scale * 0.3))
    
    def draw_hash_marks(center_pos, color):
        cx, cy = center_pos
        outer_radius = faceoff_radius + hash_distance
        
        # 4 hash marks around each circle
        angles = [0, 90, 180, 270]  # right, top, left, bottom
        for angle in angles:
            mark_x = cx + int(outer_radius * np.cos(np.radians(angle)))
            mark_y = cy + int(outer_radius * np.sin(np.radians(angle)))
            
            if angle in [0, 180]:  # horizontal marks
                start_y = mark_y - hash_length // 2
                end_y = mark_y + hash_length // 2
                cv2.line(rink_image, (mark_x, start_y), (mark_x, end_y), color, hash_thickness, cv2.LINE_AA)
            else:  # vertical marks
                start_x = mark_x - hash_length // 2
                end_x = mark_x + hash_length // 2
                cv2.line(rink_image, (start_x, mark_y), (end_x, mark_y), color, hash_thickness, cv2.LINE_AA)
    
    # Add hash marks to end zone circles only
    for fx, fy in [(left_faceoff_x, faceoff_y1), (left_faceoff_x, faceoff_y2),
                   (right_faceoff_x, faceoff_y1), (right_faceoff_x, faceoff_y2)]:
        draw_hash_marks((fx, fy), red_line_color)
    
    # Goals - simple line representation
    goal_width = int(6 * scale)
    goal_thickness = max(2, int(scale * 0.6))
    
    goal_top = center_y - goal_width // 2
    goal_bottom = center_y + goal_width // 2
    
    # Goal lines
    cv2.line(rink_image, (left_goal_line, goal_top), (left_goal_line, goal_bottom), 
             red_line_color, goal_thickness, cv2.LINE_AA)
    cv2.line(rink_image, (right_goal_line, goal_top), (right_goal_line, goal_bottom), 
             red_line_color, goal_thickness, cv2.LINE_AA)
    
    # Goal posts (small circles)
    post_radius = max(2, int(scale * 0.4))
    cv2.circle(rink_image, (left_goal_line, goal_top), post_radius, red_line_color, -1, cv2.LINE_AA)
    cv2.circle(rink_image, (left_goal_line, goal_bottom), post_radius, red_line_color, -1, cv2.LINE_AA)
    cv2.circle(rink_image, (right_goal_line, goal_top), post_radius, red_line_color, -1, cv2.LINE_AA)
    cv2.circle(rink_image, (right_goal_line, goal_bottom), post_radius, red_line_color, -1, cv2.LINE_AA)
    
    # Referee's crease (small semicircle at center bottom)
    ref_crease_radius = int(3 * scale)
    if ref_crease_radius > 0:
        cv2.ellipse(rink_image, (center_x, rink_bottom - board_thickness), 
                   (ref_crease_radius, ref_crease_radius), 0, 0, 180, 
                   red_line_color, max(1, circle_thickness // 2), cv2.LINE_AA)
    
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
        
        # Use fixed NHL dimensions to match draw_rink function
        self.rink_length_ft = 200  # feet
        self.rink_width_ft = 85    # feet
        
        # Default team colors
        self.team_colors = team_colors or {
            0: (147, 20, 255),   # Deep pink for team 0
            1: (255, 191, 0),    # Blue for team 1
            2: (71, 99, 255)     # Red for goalies
        }
        
        # Create base rink image using the new professional draw_rink function
        self.base_rink = draw_rink(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=2
        )
        
        # Calculate rink dimensions for coordinate mapping using NHL dimensions
        self.rink_width = int(self.rink_width_ft * scale)
        self.rink_length = int(self.rink_length_ft * scale)
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
        
        # Calculate homography from keypoints if available
        if keypoints and len(keypoints) >= 4:
            # Only recalculate if we don't have a homography yet
            if self.homography is None:
                # Create a temporary detector to calculate homography
                from common.rink_keypoint_detector import RinkKeypointDetector
                detector = RinkKeypointDetector(None)  # Don't need model for homography calc
                homography = detector.get_rink_homography(keypoints)
                if homography is not None:
                    self.set_homography(homography)
                    print(f"Homography matrix set successfully")
    
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
            
            # Clamp to valid rink bounds for safety
            map_x = max(0, min(map_x, self.rink_length + 2 * self.padding))
            map_y = max(0, min(map_y, self.rink_width + 2 * self.padding))
            
            return int(map_x), int(map_y)
        else:
            # Fallback: Use adaptive mapper based on camera view
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