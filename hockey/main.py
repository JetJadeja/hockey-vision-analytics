import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# (This function is generic and can be copied from the soccer example)
def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


# --- Hockey Specific Imports ---
# Assuming your new files are in sports/annotators/ and sports/configs/
from sports.annotators.soccer import draw_points_on_pitch # Reusable
from sports.common.ball import BallTracker as PuckTracker, BallAnnotator as PuckAnnotator # Aliasing for clarity
from sports.common.team import TeamClassifier
# NOTE: ViewTransformer and pitch-related functions are for the RADAR view. We'll disable that for now.

# --- UPDATE PATHS AND CONSTANTS ---
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/hockey-player-detection.pt')
PUCK_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/hockey-puck-detection.pt')
# RINK_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/hockey-rink-detection.pt') # For later

# IMPORTANT: Update these class IDs to match your custom player detection model
GOALKEEPER_CLASS_ID = 0
PLAYER_CLASS_ID = 1
REFEREE_CLASS_ID = 2

# (This function is from the soccer example and can be reused as is)
def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    if len(players) == 0 or len(goalkeepers) == 0:
        # Cannot determine team if no players are on ice, default to team 0
        return np.zeros(len(goalkeepers), dtype=int)
        
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    team_0_mask = players_team_id == 0
    team_1_mask = players_team_id == 1
    
    if not np.any(team_0_mask) or not np.any(team_1_mask):
        # If only one team is detected, assign goalie to that team.
        # Fallback to team 0 if something is wrong.
        team_id = 0 if np.any(team_0_mask) else (1 if np.any(team_1_mask) else 0)
        return np.full(len(goalkeepers), team_id, dtype=int)

    team_0_centroid = players_xy[team_0_mask].mean(axis=0)
    team_1_centroid = players_xy[team_1_mask].mean(axis=0)
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


# --- Annotators (can be copied from soccer example) ---
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700'] # Team1, Team2, Referee, Puck
BOX_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(COLORS), text_color=sv.Color.WHITE, text_padding=5)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.WHITE,
    text_padding=5,
    text_position=sv.Position.BOTTOM_CENTER,
)

class Mode(Enum):
    """
    Enum class representing the different modes of operation.
    PITCH_DETECTION and RADAR are commented out until the rink keypoint model is ready.
    """
    # PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    PUCK_DETECTION = 'PUCK_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    # RADAR = 'RADAR'

# --- PROCESSING FUNCTIONS ---

def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    for frame in tqdm(frame_generator, desc="Player Detection"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # You might need to add logic here to map your model's class_ids if they differ
        
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

def run_puck_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    puck_detection_model = YOLO(PUCK_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    puck_tracker = PuckTracker(buffer_size=20)
    puck_annotator = PuckAnnotator(radius=10, buffer_size=10, color_palette=sv.ColorPalette.from_hex(['#FFD700']))

    # InferenceSlicer is great for small objects like pucks
    slicer = sv.InferenceSlicer(
        callback=lambda image_slice: sv.Detections.from_ultralytics(puck_detection_model(image_slice, imgsz=640, verbose=False)[0]),
        slice_wh=(640, 640),
        overlap_filter_strategy=sv.OverlapFilter.NONE
    )

    for frame in tqdm(frame_generator, desc="Puck Detection"):
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = puck_tracker.update(detections)
        
        annotated_frame = frame.copy()
        annotated_frame = puck_annotator.annotate(annotated_frame, detections)
        yield annotated_frame
        
def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    for frame in tqdm(frame_generator, desc="Player Tracking"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
        yield annotated_frame

def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    
    # First pass: collect crops to train the team classifier
    frame_generator_for_crops = sv.get_video_frames_generator(source_path=source_video_path, stride=10)
    crops = []
    for frame in tqdm(frame_generator_for_crops, desc='Collecting crops for team classification'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        # Filter for players only, based on your class ID
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        crops.extend(get_crops(frame, player_detections))

    if not crops:
        print("No players detected to train team classifier. Exiting.")
        return

    # Fit the classifier
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    print("Team classifier trained successfully.")

    # Second pass: apply the classifier
    frame_generator_for_tracking = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in tqdm(frame_generator_for_tracking, desc="Team Classification"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        
        player_crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(player_crops)

        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

        # Merge detections and create color lookup
        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


# --- Main Execution Block ---

def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    frame_generator = None
    
    if mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.PUCK_DETECTION:
        frame_generator = run_puck_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(source_video_path=source_video_path, device=device)
    # The modes below are disabled until you have a rink keypoint detection model
    # elif mode == Mode.PITCH_DETECTION:
    #     raise NotImplementedError("PITCH_DETECTION mode requires a trained rink keypoint detection model.")
    # elif mode == Mode.RADAR:
    #     raise NotImplementedError("RADAR mode requires a trained rink keypoint detection model.")
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented or currently disabled.")

    if frame_generator:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in frame_generator:
                sink.write_frame(frame)
                # Optional: display the frame live
                cv2.imshow("Hockey Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hockey Video Analysis')
    parser.add_argument('--source_video_path', type=str, required=True, help='Path to the source hockey video.')
    parser.add_argument('--target_video_path', type=str, required=True, help='Path to save the annotated video.')
    parser.add_argument('--device', type=str, default='cpu', help="Device to run models on ('cpu', 'cuda', 'mps').")
    # Set a default mode that you know works
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_TRACKING, choices=list(Mode), help='The analysis mode to run.')
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )