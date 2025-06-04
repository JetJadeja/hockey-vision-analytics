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
   - `common/team.py` - TeamClassifier with hybrid deep learning + color analysis approach
   - `common/team_hybrid.py` - Advanced hybrid classifier using MobileNet + color features + spectral clustering
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
- Team classification uses hybrid approach: MobileNet deep features + color analysis + spectral clustering

## Dependencies
The project uses:
- ultralytics (YOLO)
- supervision (tracking, annotation)
- easyocr (jersey number detection)
- opencv-python, numpy, torch, scikit-learn
- torchvision (for MobileNet in hybrid team classification)

Note: No requirements.txt exists - dependencies must be installed manually.

## Team Classification Details
The hybrid classifier (`team_hybrid.py`) combines:
1. Deep features from pre-trained MobileNetV3 (captures visual patterns)
2. Color features (HSV/LAB histograms, saturation analysis, white detection)
3. Spectral clustering for robust team separation
4. Temporal consistency using player tracking history
5. Automatic fallback to simple white vs. colored detection if needed

## Environment Variables
The project uses a `.env` file containing:
- `ROBOFLOW_API_KEY` - Used for dataset access (currently set)