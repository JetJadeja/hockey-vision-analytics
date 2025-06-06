import cv2
import supervision as sv
import numpy as np

from hockey.configs.hockey import HockeyRinkConfiguration

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

if __name__ == "__main__":
    # Create a standard hockey rink configuration
    config = HockeyRinkConfiguration()  # Use default values

    # Draw the rink with better scaling
    rink_image = draw_rink(config, scale=3.0, padding=30)

    # Display the image using OpenCV
    cv2.imshow("Hockey Rink", rink_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()