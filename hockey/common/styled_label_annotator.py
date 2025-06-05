import cv2
import numpy as np
import supervision as sv
from typing import List, Optional, Union


class StyledLabelAnnotator:
    """
    Custom label annotator with improved styling for hockey player labels.
    """
    
    def __init__(
        self,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        padding: int = 4,
        border_radius: int = 4,
        opacity: float = 0.7
    ):
        """
        Initialize the styled label annotator.
        
        Args:
            font_scale: Scale factor for the font size
            font_thickness: Thickness of the font
            padding: Padding around the text
            border_radius: Radius for rounded corners
            opacity: Background opacity (0-1)
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.padding = padding
        self.border_radius = border_radius
        self.opacity = opacity
    
    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        labels: List[str],
        custom_color_lookup: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Annotate the scene with styled labels.
        
        Args:
            scene: The frame to annotate
            detections: Detection results
            labels: List of labels for each detection
            custom_color_lookup: Optional array mapping detections to color indices
            
        Returns:
            Annotated frame
        """
        annotated_scene = scene.copy()
        
        # Define colors for teams and goalies - matching COLORS from main.py
        colors = [
            (147, 20, 255),   # #FF1493 Deep pink for team 0
            (255, 191, 0),    # #00BFFF Deep sky blue for team 1  
            (71, 99, 255)     # #FF6347 Tomato red for goalies
        ]
        
        for i, (xyxy, label) in enumerate(zip(detections.xyxy, labels)):
            if not label:
                continue
                
            # Get color based on custom lookup or default
            if custom_color_lookup is not None and i < len(custom_color_lookup):
                color_idx = int(custom_color_lookup[i])
                color = colors[color_idx % len(colors)]
            else:
                color = colors[0]
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            # Position label above the detection box
            x1, y1, x2, y2 = xyxy.astype(int)
            
            # Calculate label position (centered above box)
            label_x = int((x1 + x2) / 2 - text_width / 2)
            label_y = y1 - self.padding - baseline
            
            # Ensure label is within frame bounds
            label_x = max(self.padding, label_x)
            label_x = min(scene.shape[1] - text_width - self.padding, label_x)
            label_y = max(text_height + self.padding, label_y)
            
            # Draw simple rectangular background
            bg_x1 = label_x - self.padding
            bg_y1 = label_y - text_height - self.padding
            bg_x2 = label_x + text_width + self.padding
            bg_y2 = label_y + baseline + self.padding
            
            # Draw filled rectangle with team color
            cv2.rectangle(
                annotated_scene,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                color,
                -1  # Filled
            )
            
            # Draw text in white
            cv2.putText(
                annotated_scene,
                label,
                (label_x, label_y),
                self.font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness,
                cv2.LINE_AA
            )
        
        return annotated_scene
    
    def _draw_rounded_rectangle(
        self,
        img: np.ndarray,
        pt1: tuple,
        pt2: tuple,
        color: tuple,
        radius: int,
        thickness: int
    ):
        """
        Draw a rounded rectangle on the image.
        
        Args:
            img: Image to draw on
            pt1: Top-left corner
            pt2: Bottom-right corner
            color: BGR color
            radius: Corner radius
            thickness: Line thickness (-1 for filled)
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Ensure radius is not too large
        radius = min(radius, int(min(x2 - x1, y2 - y1) / 2))
        
        # Draw the rectangles and circles for rounded corners
        # Top rectangle
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        # Left rectangle
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)