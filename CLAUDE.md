# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hockey vision analytics system that uses computer vision and deep learning to analyze hockey footage. The system detects players, tracks the puck, recognizes jersey numbers, and classifies teams.

## Key Commands

### Running the Application
```bash
python hockey/main.py --source_path <input_video> --target_path <output_video> --mode <MODE> --device <cuda|cpu>
```

Available modes:
- `PLAYER_DETECTION` - Detect players/goalies with jersey numbers
- `PUCK_DETECTION` - Track puck with visual trail
- `PLAYER_TRACKING` - Track players across frames with persistent IDs
- `TEAM_CLASSIFICATION` - Auto-classify teams and apply color coding

### Training Models
The project uses YOLO models. Training is done via Jupyter notebooks:
- `notebooks/train_player_detection.ipynb` - Train player/goalie detection
- `notebooks/train_puck_detection.ipynb` - Train puck detection

## Architecture

### Core Processing Pipeline
The main entry point is `hockey/main.py` which orchestrates four processing modes. Each mode uses different combinations of:

1. **Detection Models** (YOLO-based):
   - Player detection model (2 classes: player, goalie)
   - Puck detection model (1 class with specialized tracking)

2. **Tracking & Classification**:
   - ByteTrack for multi-object tracking
   - SIGLIP vision embeddings + UMAP + KMeans for team classification
   - EasyOCR for jersey number recognition

3. **Key Components**:
   - `common/puck.py` - PuckTracker maintains detection history and smooths trajectories
   - `common/team.py` - TeamClassifier uses vision transformers to classify players into teams
   - `common/view.py` - ViewTransformer handles perspective transformations (not currently used)
   - `configs/hockey.py` - NHL rink dimensions and homography points

### Model Locations
Pre-trained models are stored in `hockey/data/`:
- `hockey-player-detection.pt`
- `hockey-puck-detection.pt`

### Important Implementation Details
- Player class ID = 0, Goalkeeper class ID = 1 (from 2-class model)
- Puck detection uses inference slicing for better accuracy on small objects
- Jersey numbers are persisted across frames using a dictionary keyed by tracker ID
- Team classification extracts visual embeddings from player crops and clusters them

## Dependencies
The project uses:
- ultralytics (YOLO)
- supervision (tracking, annotation)
- transformers (SIGLIP for team classification)
- easyocr (jersey number detection)
- opencv-python, numpy, torch, scikit-learn, umap-learn

Note: No requirements.txt exists - dependencies must be installed manually.

## Environment Variables
The project uses a `.env` file containing:
- `ROBOFLOW_API_KEY` - Used for dataset access (currently set)