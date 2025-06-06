import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import os

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Import hockey-specific modules
from common.team import TeamClassifier
from common.smooth_annotator import SmoothAnnotator
from common.team_selector import InteractiveTeamSelector
from common.rink_keypoint_detector import RinkKeypointDetector


@dataclass
class Config:
    """Configuration for hockey video processing"""
    # Model paths
    player_model_name: str = 'hockey-player-detection.pt'
    hockey_model_name: str = 'hockey-detection.pt'
    
    # Detection parameters
    detection_imgsz: int = 1280
    detection_confidence: float = 0.4
    
    # Tracking parameters
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30
    minimum_consecutive_frames: int = 2
    
    # Team classification
    initialization_stride: int = 10
    max_initialization_frames: int = 20
    min_players_for_selection: int = 6
    
    # Annotation
    smoothing_factor: float = 0.3
    use_adaptive_smoothing: bool = True
    
    # Visualization
    team_colors: List[str] = None
    annotation_thickness: int = 2
    label_text_scale: float = 0.6
    label_text_thickness: int = 2
    
    # Rink keypoint detection
    keypoint_confidence_threshold: float = 0.3
    keypoint_radius: int = 10
    
    def __post_init__(self):
        if self.team_colors is None:
            self.team_colors = ['#FF1493', '#00BFFF', '#FF6347']  # Team1, Team2, Goalies


class ModelManager:
    """Manages loading and validation of models"""
    
    def __init__(self, data_dir: Path, config: Config):
        self.data_dir = data_dir
        self.config = config
        self.player_model = None
        self.rink_detector = None
    
    def load_player_model(self, device: str) -> YOLO:
        """Load and validate player detection model"""
        model_path = self.data_dir / self.config.player_model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Player detection model not found: {model_path}")
        
        self.player_model = YOLO(str(model_path)).to(device=device)
        return self.player_model
    
    def load_rink_detector(self) -> Optional[RinkKeypointDetector]:
        """Load rink keypoint detector if needed"""
        model_path = self.data_dir / self.config.hockey_model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Hockey detection model not found: {model_path}")
        
        self.rink_detector = RinkKeypointDetector(str(model_path))
        return self.rink_detector


class AnnotationManager:
    """Manages all annotation and visualization components"""
    
    def __init__(self, config: Config):
        self.config = config
        color_palette = sv.ColorPalette.from_hex(config.team_colors)
        
        # Initialize annotators
        base_annotator = sv.BoxAnnotator(
            color=color_palette,
            thickness=config.annotation_thickness
        )
        
        self.box_annotator = SmoothAnnotator(
            base_annotator,
            smoothing_factor=config.smoothing_factor,
            use_adaptive=config.use_adaptive_smoothing
        )
        
        self.label_annotator = sv.LabelAnnotator(
            color=color_palette,
            text_color=sv.Color.from_hex('#FFFFFF'),
            text_padding=5,
            text_scale=config.label_text_scale,
            text_thickness=config.label_text_thickness
        )
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: List[str],
        color_lookup: np.ndarray,
        rink_keypoints: Optional[List] = None
    ) -> np.ndarray:
        """Apply all annotations to frame"""
        annotated = frame.copy()
        
        # Apply box and label annotations
        annotated = self.box_annotator.annotate(
            annotated, detections, custom_color_lookup=color_lookup
        )
        annotated = self.label_annotator.annotate(
            annotated, detections, labels, custom_color_lookup=color_lookup
        )
        
        # Apply rink keypoints if available
        if rink_keypoints:
            # This would be handled by rink_detector.visualize_keypoints
            pass
        
        return annotated


class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, config: Config, device: str, enable_rink_keypoints: bool = False):
        self.config = config
        self.device = device
        self.enable_rink_keypoints = enable_rink_keypoints
        
        # Get paths
        self.data_dir = Path(__file__).parent / 'data'
        
        # Initialize components
        self.model_manager = ModelManager(self.data_dir, config)
        self.annotation_manager = AnnotationManager(config)
        self.team_classifier = TeamClassifier(device=device)
        self.team_selector = InteractiveTeamSelector()
        
        # Initialize tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
            minimum_consecutive_frames=config.minimum_consecutive_frames
        )
        
        # Load models
        self.player_model = self.model_manager.load_player_model(device)
        self.rink_detector = None
        if enable_rink_keypoints:
            self.rink_detector = self.model_manager.load_rink_detector()
            print("Rink keypoint detection enabled")
    
    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """Run player detection on frame"""
        result = self.player_model(
            frame,
            imgsz=self.config.detection_imgsz,
            verbose=False,
            conf=self.config.detection_confidence
        )[0]
        
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for valid classes and confidence
        valid_detections = detections[
            ((detections.class_id == PLAYER_CLASS_ID) | 
             (detections.class_id == GOALKEEPER_CLASS_ID)) &
            (detections.confidence > self.config.detection_confidence)
        ]
        
        return valid_detections
    
    def initialize_team_classifier(self, source_path: str) -> None:
        """Initialize team classifier with sample frames"""
        print("Initializing team classification...")
        
        crops = []
        positions = []
        first_frame = None
        first_tracked_detections = None
        
        # Create temporary tracker for initialization
        temp_tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            minimum_consecutive_frames=1,
            frame_rate=self.config.frame_rate
        )
        
        # Sample frames for initialization
        frame_generator = sv.get_video_frames_generator(
            source_path=source_path,
            stride=self.config.initialization_stride
        )
        
        for i, frame in enumerate(frame_generator):
            if i > self.config.max_initialization_frames:
                break
            
            # Detect players
            detections = self.detect_players(frame)
            player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
            
            # Track detections
            tracked_dets = temp_tracker.update_with_detections(player_detections)
            
            # Look for frame with enough players
            if first_frame is None and len(tracked_dets) >= self.config.min_players_for_selection:
                first_frame = frame
                first_tracked_detections = tracked_dets
            
            # Collect crops and positions
            crops.extend(self._get_crops(frame, player_detections))
            positions.extend(self._get_positions(player_detections))
        
        # Interactive team selection
        team_selection = None
        if first_frame is not None and first_tracked_detections is not None:
            team_selection = self.team_selector.select_teams(first_frame, first_tracked_detections)
        
        if team_selection:
            self.team_classifier.set_team_names(team_selection.team_names)
            print(f"Teams set: {team_selection.team_names[0]} vs {team_selection.team_names[1]}")
        else:
            print("Team selection cancelled, using default team names")
        
        # Fit classifier
        self.team_classifier.fit(
            crops,
            positions=positions,
            frame=first_frame,
            detections=first_tracked_detections
        )
        print("Classifier fitted.")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        # Detect players
        detections = self.detect_players(frame)
        
        # Update tracker
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Separate players and goalies
        player_detections = tracked_detections[tracked_detections.class_id == PLAYER_CLASS_ID]
        goalie_detections = tracked_detections[tracked_detections.class_id == GOALKEEPER_CLASS_ID]
        
        # Classify teams for players
        player_team_ids = np.array([])
        if len(player_detections) > 0:
            player_crops = self._get_crops(frame, player_detections)
            player_positions = self._get_positions(player_detections)
            
            player_team_ids = self.team_classifier.predict(
                player_crops,
                tracker_ids=player_detections.tracker_id,
                positions=player_positions
            )
        
        # Assign goalies to team 2
        goalie_team_ids = np.array([2] * len(goalie_detections), dtype=np.int32)
        
        # Combine detections and create color lookup
        all_detections = sv.Detections.merge([player_detections, goalie_detections])
        color_lookup = self._create_color_lookup(player_team_ids, goalie_team_ids)
        
        # Create labels
        labels = self._create_labels(all_detections, player_team_ids)
        
        # Detect rink keypoints if enabled
        rink_keypoints = None
        if self.rink_detector:
            keypoints = self.rink_detector.detect_keypoints(
                frame,
                conf_threshold=self.config.keypoint_confidence_threshold
            )
            if keypoints:
                frame = self.rink_detector.visualize_keypoints(
                    frame,
                    keypoints,
                    radius=self.config.keypoint_radius,
                    show_labels=True
                )
        
        # Annotate frame
        annotated_frame = self.annotation_manager.annotate_frame(
            frame, all_detections, labels, color_lookup, rink_keypoints
        )
        
        return annotated_frame
    
    def process_video(self, source_path: str) -> Iterator[np.ndarray]:
        """Process entire video"""
        # Initialize team classifier
        self.initialize_team_classifier(source_path)
        
        # Process frames
        for frame in sv.get_video_frames_generator(source_path=source_path):
            yield self.process_frame(frame)
    
    def _get_crops(self, frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
        """Extract crops from detections"""
        return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    
    def _get_positions(self, detections: sv.Detections) -> List[Tuple[float, float]]:
        """Extract center positions from detections"""
        positions = []
        for xyxy in detections.xyxy:
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            positions.append((center_x, center_y))
        return positions
    
    def _create_color_lookup(self, player_team_ids: np.ndarray, goalie_team_ids: np.ndarray) -> np.ndarray:
        """Create color lookup array"""
        if len(player_team_ids) > 0:
            return np.concatenate([player_team_ids, goalie_team_ids]).astype(np.int32)
        return goalie_team_ids.astype(np.int32)
    
    def _create_labels(self, detections: sv.Detections, player_team_ids: np.ndarray) -> List[str]:
        """Create labels for detections"""
        labels = []
        for i, (tracker_id, class_id) in enumerate(zip(detections.tracker_id, detections.class_id)):
            if class_id == PLAYER_CLASS_ID and i < len(player_team_ids):
                team_name = self.team_classifier.get_team_name(player_team_ids[i])
                labels.append(team_name)
            elif class_id == GOALKEEPER_CLASS_ID:
                labels.append("Goalie")
            else:
                labels.append("Player")
        return labels


# Class IDs
PLAYER_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1


def process_video_with_display(
    processor: VideoProcessor,
    source_path: str,
    target_path: Optional[str] = None
) -> None:
    """Process video with optional display and saving"""
    frame_generator = processor.process_video(source_path)
    
    if target_path:
        video_info = sv.VideoInfo.from_video_path(source_path)
        with sv.VideoSink(target_path, video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                sink.write_frame(frame)
                cv2.imshow("Hockey Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    else:
        for frame in frame_generator:
            cv2.imshow("Hockey Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Hockey Vision Analytics')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source video file.')
    parser.add_argument('--target_path', type=str, default=None, help='Path to save the output video.')
    parser.add_argument('--device', type=str, default='cpu', help="Device to run models on ('cpu', 'cuda', 'mps').")
    parser.add_argument('--rink-keypoints', action='store_true', help='Enable rink keypoint detection.')
    
    args = parser.parse_args()
    
    # Validate source path
    if not Path(args.source_path).exists():
        raise FileNotFoundError(f"Source video not found: {args.source_path}")
    
    # Create configuration
    config = Config()
    
    # Create processor
    processor = VideoProcessor(
        config=config,
        device=args.device,
        enable_rink_keypoints=args.rink_keypoints
    )
    
    # Process video
    process_video_with_display(
        processor=processor,
        source_path=args.source_path,
        target_path=args.target_path
    )


if __name__ == '__main__':
    main()