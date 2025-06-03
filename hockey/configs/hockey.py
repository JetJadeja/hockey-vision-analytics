from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class HockeyRinkConfiguration:
    """
    Configuration for a standard North American (NHL-style) hockey rink.
    Dimensions are in feet, but represented as integers for simplicity in calculations.
    """
    width: int = 85  # [ft]
    length: int = 200 # [ft]
    goal_line_from_end: int = 11 # [ft]
    blue_line_from_center: int = 25 # [ft]
    faceoff_circle_radius: int = 15 # [ft]
    faceoff_dot_to_goal_line: int = 20 # [ft]
    faceoff_dot_from_side: int = 22 # [ft]
    
    # Additional parameters for neutral zone dots and hash marks
    neutral_zone_dot_from_center: int = 22 # [ft] Distance of neutral zone dots from center line
    neutral_zone_dot_from_side: int = 20 # [ft] Distance from side boards
    faceoff_hash_mark_length: int = 2 # [ft] Length of hash marks around faceoff circles
    faceoff_hash_mark_distance: int = 2 # [ft] Distance of hash marks from circle

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """
        Returns a list of key points (vertices) for the hockey rink.
        These points are crucial for the homography transformation.
        We need to expand this list to include all the key points
        our rink detection model can identify (corners, goal lines, blue lines, etc.).
        """
        center_x = self.length / 2
        center_y = self.width / 2
        
        return [
            # Rink corners
            (0, 0),  # 1: Top-left
            (self.length, 0),  # 2: Top-right
            (self.length, self.width),  # 3: Bottom-right
            (0, self.width),  # 4: Bottom-left

            # Center line points
            (center_x, 0), # 5: Center line top
            (center_x, self.width), # 6: Center line bottom
            
            # Blue lines top
            (center_x - self.blue_line_from_center, 0), # 7
            (center_x + self.blue_line_from_center, 0), # 8

            # Blue lines bottom
            (center_x - self.blue_line_from_center, self.width), # 9
            (center_x + self.blue_line_from_center, self.width), # 10

            # Goal lines top
            (self.goal_line_from_end, 0), # 11
            (self.length - self.goal_line_from_end, 0), # 12

            # Goal lines bottom
            (self.goal_line_from_end, self.width), # 13
            (self.length - self.goal_line_from_end, self.width), # 14
            
            # Faceoff dots... (you can add more points)
        ]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """
        Defines the lines to be drawn on the rink based on vertices.
        This will need to be manually mapped based on your `vertices` list.
        The indices are 1-based.
        """
        return [
            (1, 2), (2, 3), (3, 4), (4, 1), # Rink boundary
            (5, 6), # Center red line
            (7, 9), # Left blue line
            (8, 10), # Right blue line
            (11, 13), # Left goal line
            (12, 14), # Right goal line
        ]

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FFD700" for _ in range(14) # Example color
    ])