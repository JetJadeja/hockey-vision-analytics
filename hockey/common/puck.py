from collections import deque
import cv2
import numpy as np
import supervision as sv

class PuckAnnotator:
    """Annotates frames with circles to show puck trail."""
    def __init__(self, radius: int, buffer_size: int = 15, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib('winter', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i <= 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER).astype(int)
        self.buffer.append(xy)

        for i, frame_xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in frame_xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame

class PuckTracker:
    """Tracks a hockey puck by selecting the detection closest to recent positions."""
    def __init__(self, buffer_size: int = 5):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)
        
        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]