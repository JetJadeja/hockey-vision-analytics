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

# --- Constants and Paths ---
# Assumes your models are in a 'data' folder next to your 'hockey' package
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, 'data')
PLAYER_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'hockey-player-detection.pt')
PUCK_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'hockey-puck-detection.pt')

# !! IMPORTANT !!: Update these class IDs to match your trained models
# Player detection model classes
PLAYER_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
REFEREE_CLASS_ID = 2
# Puck detection model class
PUCK_CLASS_ID = 0

# --- Annotators ---
COLORS = ['#FF1493', '#00BFFF', '#FF6347'] # Team1, Team2, Referee
BOX_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#000000'),
    text_padding=2
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
        
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

def run_puck_detection(source_path: str, device: str) -> Iterator[np.ndarray]:
    puck_model = YOLO(PUCK_DETECTION_MODEL_PATH).to(device=device)
    puck_tracker = PuckTracker()
    puck_annotator = PuckAnnotator(radius=5, buffer_size=15)
    
    slicer = sv.InferenceSlicer(
        callback=lambda image_slice: sv.Detections.from_ultralytics(
            puck_model(image_slice, imgsz=640, verbose=False)[0]
        ),
        slice_wh=(640, 640),
        overlap_filter_strategy=sv.OverlapFilter.NONE
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
        detections = tracker.update_with_detections(detections)
        
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
        yield annotated_frame
        
def run_team_classification(source_path: str, device: str) -> Iterator[np.ndarray]:
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    team_classifier = TeamClassifier(device=device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=5)
    
    # First pass: collect player crops to train classifier
    print("Collecting player crops for team classification...")
    crops = []
    frame_generator = sv.get_video_frames_generator(source_path=source_path, stride=10)
    for frame in tqdm(frame_generator, desc='Collecting crops'):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        crops.extend(get_crops(frame, player_detections))

    print("Fitting team classifier...")
    team_classifier.fit(crops)
    print("Classifier fitted.")

    # Second pass: process video and apply team colors
    for frame in sv.get_video_frames_generator(source_path=source_path):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        
        # Classify teams for players
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        player_crops = get_crops(frame, player_detections)
        player_team_ids = team_classifier.predict(player_crops)

        # Handle referees (optional, assuming they have their own class)
        referee_detections = detections[detections.class_id == REFEREE_CLASS_ID]
        referee_team_ids = [REFEREE_CLASS_ID] * len(referee_detections)
        
        # Combine detections and color lookups
        all_detections = sv.Detections.merge([player_detections, referee_detections])
        color_lookup = np.array(player_team_ids.tolist() + referee_team_ids)
        
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
        
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, all_detections, custom_color_lookup=color_lookup)
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
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_TRACKING, choices=list(Mode), help='The processing mode.')
    parser.add_argument('--device', type=str, default='cpu', help="Device to run models on ('cpu', 'cuda', 'mps').")
    
    args = parser.parse_args()
    main(
        source_path=args.source_path,
        target_path=args.target_path,
        mode=args.mode,
        device=args.device
    )