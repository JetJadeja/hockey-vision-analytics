import argparse
from enum import Enum
from typing import Iterator, List
import os
import re

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import easyocr

# Import your hockey-specific modules
from common.puck import PuckTracker, PuckAnnotator
from common.team import TeamClassifier

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
BOX_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#000000'),
    text_padding=2
)

# Initialize OCR reader for jersey number detection
OCR_READER = None

def get_ocr_reader():
    global OCR_READER
    if OCR_READER is None:
        OCR_READER = easyocr.Reader(['en'], gpu=False)
    return OCR_READER

class Mode(Enum):
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    PUCK_DETECTION = 'PUCK_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def detect_jersey_number(player_crop: np.ndarray) -> str:
    """
    Detect jersey number from player crop using OCR.
    Returns the detected number or 'none' if no valid number found.
    """
    try:
        # Resize crop for better OCR
        height, width = player_crop.shape[:2]
        if height < 100 or width < 50:
            return "none"
        
        # Focus on the upper body area where numbers typically are
        upper_crop = player_crop[:int(height * 0.7), :]
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to make text more visible
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR detection
        reader = get_ocr_reader()
        results = reader.readtext(thresh, allowlist='0123456789')
        
        # Filter for valid jersey numbers (1-99)
        valid_numbers = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Confidence threshold
                # Extract numbers only
                numbers = re.findall(r'\d+', text)
                for num in numbers:
                    if 1 <= int(num) <= 99:  # Valid jersey number range
                        valid_numbers.append(num)
        
        if valid_numbers:
            # Return the most likely number (first one with highest confidence)
            return valid_numbers[0]
        else:
            return "none"
            
    except Exception:
        return "none"

def get_jersey_numbers(frame: np.ndarray, detections: sv.Detections) -> List[str]:
    """Get jersey numbers for all detected players."""
    if len(detections) == 0:
        return []
    
    crops = get_crops(frame, detections)
    numbers = []
    
    for crop in crops:
        number = detect_jersey_number(crop)
        numbers.append(number)
    
    return numbers

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
        
        # Get jersey numbers
        jersey_numbers = get_jersey_numbers(frame, valid_detections)
        labels = [f"#{num}" if num != "none" else "none" for num in jersey_numbers]
        
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, valid_detections)
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
    
    # Dictionary to store jersey numbers for each tracker ID
    tracker_jersey_numbers = {}
    
    for frame in sv.get_video_frames_generator(source_path=source_path):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for valid classes only (player and goalie)
        valid_detections = detections[
            (detections.class_id == PLAYER_CLASS_ID) | 
            (detections.class_id == GOALKEEPER_CLASS_ID)
        ]
        
        tracked_detections = tracker.update_with_detections(valid_detections)
        
        # Get jersey numbers for current detections
        jersey_numbers = get_jersey_numbers(frame, tracked_detections)
        
        # Update tracker jersey number mapping
        for i, tracker_id in enumerate(tracked_detections.tracker_id):
            if i < len(jersey_numbers) and jersey_numbers[i] != "none":
                # If we detect a number, update the mapping
                tracker_jersey_numbers[tracker_id] = jersey_numbers[i]
            elif tracker_id not in tracker_jersey_numbers:
                # If no number detected and no previous mapping, set to none
                tracker_jersey_numbers[tracker_id] = "none"
        
        # Create labels using stored jersey numbers
        labels = []
        for tracker_id in tracked_detections.tracker_id:
            jersey_num = tracker_jersey_numbers.get(tracker_id, "none")
            if jersey_num != "none":
                labels.append(f"#{jersey_num}")
            else:
                labels.append("none")
        
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, tracked_detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, tracked_detections, labels=labels)
        yield annotated_frame
        
def run_team_classification(source_path: str, device: str) -> Iterator[np.ndarray]:
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    team_classifier = TeamClassifier(device=device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=5)
    
    # Dictionary to store jersey numbers for each tracker ID
    tracker_jersey_numbers = {}
    
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
            # Pass tracker IDs for temporal consistency
            player_team_ids = team_classifier.predict(player_crops, tracker_ids=player_detections.tracker_id)
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
        
        # Get jersey numbers for all tracked players
        jersey_numbers = get_jersey_numbers(frame, all_detections)
        
        # Update tracker jersey number mapping
        for i, tracker_id in enumerate(all_detections.tracker_id):
            if i < len(jersey_numbers) and jersey_numbers[i] != "none":
                tracker_jersey_numbers[tracker_id] = jersey_numbers[i]
            elif tracker_id not in tracker_jersey_numbers:
                tracker_jersey_numbers[tracker_id] = "none"
        
        # Create labels using stored jersey numbers
        labels = []
        for tracker_id in all_detections.tracker_id:
            jersey_num = tracker_jersey_numbers.get(tracker_id, "none")
            if jersey_num != "none":
                labels.append(f"#{jersey_num}")
            else:
                labels.append("none")
        
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
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION, choices=list(Mode), help='The processing mode.')
    parser.add_argument('--device', type=str, default='cpu', help="Device to run models on ('cpu', 'cuda', 'mps').")
    
    args = parser.parse_args()
    main(
        source_path=args.source_path,
        target_path=args.target_path,
        mode=args.mode,
        device=args.device
    )