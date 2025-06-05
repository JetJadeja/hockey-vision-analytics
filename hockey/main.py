import argparse
from enum import Enum
from typing import Iterator, List
import os

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Import your hockey-specific modules
from common.puck import PuckTracker, PuckAnnotator
from common.team import TeamClassifier
from common.smooth_annotator import SmoothAnnotator
from common.team_selector import InteractiveTeamSelector
from common.styled_label_annotator import StyledLabelAnnotator

# --- Constants and Paths ---
# Assumes your models are in a 'data' folder next to your 'hockey' package
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, 'data')
PLAYER_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'hockey-player-detection.pt')
PUCK_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'hockey-puck-detection.pt')

# !! IMPORTANT !!: Updated class IDs to match your 2-class model (data_players_only.yaml)
# Player detection model classes
PLAYER_CLASS_ID = 0      # 'player' class  
GOALKEEPER_CLASS_ID = 1  # 'goalie' class
# Note: REFEREE_CLASS_ID removed - not in current 2-class model

# Puck detection model class
PUCK_CLASS_ID = 0

# --- Annotators ---
COLORS = ['#FF1493', '#00BFFF', '#FF6347'] # Team1, Team2, Goalies
# Create base annotators
# BASE_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
BASE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)

# Wrap with smooth annotators for stable visualization
# Higher smoothing factor = more stability, less responsiveness
ANNOTATOR = SmoothAnnotator(BASE_ANNOTATOR, smoothing_factor=0.5)
# Use supervision's label annotator with matching colors
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_scale=0.6,
    text_thickness=2
)

class Mode(Enum):
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    PUCK_DETECTION = 'PUCK_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

# --- Processing Functions ---

def run_player_detection(source_path: str, device: str) -> Iterator[np.ndarray]:
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    for frame in sv.get_video_frames_generator(source_path=source_path):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for valid classes only (player and goalie)
        valid_detections = detections[
            (detections.class_id == PLAYER_CLASS_ID) | 
            (detections.class_id == GOALKEEPER_CLASS_ID)
        ]
        
        # Create simple labels based on class
        labels = []
        for class_id in valid_detections.class_id:
            if class_id == PLAYER_CLASS_ID:
                labels.append("Player")
            elif class_id == GOALKEEPER_CLASS_ID:
                labels.append("Goalie")
            else:
                labels.append("Unknown")
        
        annotated_frame = frame.copy()
        annotated_frame = ANNOTATOR.annotate(annotated_frame, valid_detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, valid_detections, labels=labels)
        yield annotated_frame

def run_puck_detection(source_path: str, device: str) -> Iterator[np.ndarray]:
    puck_model = YOLO(PUCK_DETECTION_MODEL_PATH).to(device=device)
    puck_tracker = PuckTracker()
    puck_annotator = PuckAnnotator(radius=5, buffer_size=15)
    
    slicer = sv.InferenceSlicer(
        callback=lambda image_slice: sv.Detections.from_ultralytics(
            puck_model(image_slice, imgsz=640, verbose=False)[0]
        ),
        slice_wh=(640, 640)
    )

    for frame in sv.get_video_frames_generator(source_path=source_path):
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = puck_tracker.update(detections)
        
        annotated_frame = frame.copy()
        annotated_frame = puck_annotator.annotate(annotated_frame, detections)
        yield annotated_frame

def run_player_tracking(source_path: str, device: str) -> Iterator[np.ndarray]:
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=5)
    
    for frame in sv.get_video_frames_generator(source_path=source_path):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for valid classes only (player and goalie)
        valid_detections = detections[
            (detections.class_id == PLAYER_CLASS_ID) | 
            (detections.class_id == GOALKEEPER_CLASS_ID)
        ]
        
        tracked_detections = tracker.update_with_detections(valid_detections)
        
        # Create labels with tracker ID and class
        labels = []
        for i, (tracker_id, class_id) in enumerate(zip(tracked_detections.tracker_id, tracked_detections.class_id)):
            class_name = "Player" if class_id == PLAYER_CLASS_ID else "Goalie"
            labels.append(f"{class_name} {tracker_id}")
        
        annotated_frame = frame.copy()
        annotated_frame = ANNOTATOR.annotate(annotated_frame, tracked_detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, tracked_detections, labels=labels)
        yield annotated_frame

def run_team_classification(source_path: str, device: str) -> Iterator[np.ndarray]:
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    team_classifier = TeamClassifier(device=device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=5)
    team_selector = InteractiveTeamSelector()
    
    # Fit classifier with sample frames
    print("Initializing team classification...")
    crops = []
    positions = []
    first_frame = None
    first_detections = None
    first_tracked_detections = None
    
    # Find a good frame for interactive selection
    frame_generator = sv.get_video_frames_generator(source_path=source_path, stride=10)
    temp_tracker = sv.ByteTrack(minimum_consecutive_frames=1)
    
    for i, frame in enumerate(frame_generator):
        if i > 20:  # Just sample first 20 frames for fitting
            break
        
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(result)
        player_detections = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
        
        # Track detections for interactive selection
        tracked_dets = temp_tracker.update_with_detections(player_detections)
        
        # Look for frame with good player visibility
        if first_frame is None and len(tracked_dets) >= 6:
            first_frame = frame
            first_detections = player_detections
            first_tracked_detections = tracked_dets
        
        crops.extend(get_crops(frame, player_detections))
        
        # Extract center positions of bounding boxes
        if len(player_detections) > 0:
            for xyxy in player_detections.xyxy:
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                positions.append((center_x, center_y))
    
    # Interactive team selection
    team_selection = None
    if first_frame is not None and first_tracked_detections is not None:
        team_selection = team_selector.select_teams(first_frame, first_tracked_detections)
    
    if team_selection:
        # Set team names in classifier
        team_classifier.set_team_names(team_selection.team_names)
        print(f"Teams set: {team_selection.team_names[0]} vs {team_selection.team_names[1]}")
    else:
        print("Team selection cancelled, using default team names")
    
    team_classifier.fit(crops, positions=positions, frame=first_frame, detections=first_detections)
    print("Classifier fitted.")

    # Process video and apply team colors
    for frame in sv.get_video_frames_generator(source_path=source_path):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for valid classes only
        valid_detections = detections[
            (detections.class_id == PLAYER_CLASS_ID) | 
            (detections.class_id == GOALKEEPER_CLASS_ID)
        ]
        
        tracked_detections = tracker.update_with_detections(valid_detections)
        
        # Classify teams for players only
        player_detections = tracked_detections[tracked_detections.class_id == PLAYER_CLASS_ID]
        goalie_detections = tracked_detections[tracked_detections.class_id == GOALKEEPER_CLASS_ID]
        
        # Get team classifications for players
        if len(player_detections) > 0:
            player_crops = get_crops(frame, player_detections)
            # Extract positions for current frame
            player_positions = []
            for xyxy in player_detections.xyxy:
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                player_positions.append((center_x, center_y))
            # Pass tracker IDs and positions for better classification
            player_team_ids = team_classifier.predict(
                player_crops, 
                tracker_ids=player_detections.tracker_id,
                positions=player_positions
            )
        else:
            player_team_ids = np.array([])

        # Assign goalies to team 2 (different color)
        goalie_team_ids = np.array([2] * len(goalie_detections), dtype=np.int32)
        
        # Combine detections and color lookups
        all_detections = sv.Detections.merge([player_detections, goalie_detections])
        if len(player_team_ids) > 0:
            color_lookup = np.concatenate([player_team_ids, goalie_team_ids]).astype(np.int32)
        else:
            color_lookup = goalie_team_ids.astype(np.int32)
        
        # Create labels with team names for players and 'Goalie' for goalies
        labels = []
        for i, (tracker_id, class_id) in enumerate(zip(all_detections.tracker_id, all_detections.class_id)):
            if class_id == PLAYER_CLASS_ID and i < len(player_team_ids):
                team_name = team_classifier.get_team_name(player_team_ids[i])
                labels.append(team_name)
            elif class_id == GOALKEEPER_CLASS_ID:
                labels.append("Goalie")
            else:
                labels.append("Player")
        
        annotated_frame = frame.copy()
        
        # Apply regular annotations only (segmentation removed)
        annotated_frame = ANNOTATOR.annotate(annotated_frame, all_detections, custom_color_lookup=color_lookup)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, all_detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame

# --- Main Orchestrator ---

def main(source_path: str, target_path: str, mode: Mode, device: str):
    if mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(source_path, device)
    elif mode == Mode.PUCK_DETECTION:
        frame_generator = run_puck_detection(source_path, device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(source_path, device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(source_path, device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    if target_path:
        video_info = sv.VideoInfo.from_video_path(source_path)
        with sv.VideoSink(target_path, video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                sink.write_frame(frame)
                cv2.imshow("Hockey Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    else: # If no target path, just display
        for frame in frame_generator:
            cv2.imshow("Hockey Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hockey Vision Analytics')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source video file.')
    parser.add_argument('--target_path', type=str, default=None, help='Path to save the output video.')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION, choices=list(Mode), help='The processing mode.')
    parser.add_argument('--device', type=str, default='cpu', help="Device to run models on ('cpu', 'cuda', 'mps').")
    
    args = parser.parse_args()
    main(
        source_path=args.source_path,
        target_path=args.target_path,
        mode=args.mode,
        device=args.device
    )