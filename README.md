# Hockey Vision Analytics

A comprehensive computer vision system for analyzing hockey footage using deep learning. The system detects players, tracks the puck, recognizes jersey numbers, classifies teams, and provides real-time 2D rink visualization with player positions.

## 🏒 Overview

This project leverages state-of-the-art computer vision techniques to extract meaningful insights from hockey video footage. It combines object detection, tracking, OCR, and homography transformation to create a complete analytics pipeline.

### Key Features

- **Player Detection & Tracking**: Identifies players and goalies with persistent tracking across frames
- **Team Classification**: Automatically classifies players into teams using visual features
- **Puck Detection & Tracking**: Tracks the puck with visual trail visualization
- **Jersey Number Recognition**: Reads player jersey numbers using OCR
- **2D Rink Visualization**: Maps player positions from video to a 2D overhead rink view
- **Interactive Calibration**: Manual calibration system for accurate position mapping
- **Multi-mode Processing**: Different visualization modes for various analytics needs

## 🛠️ Technical Architecture

### System Components

#### 1. **Detection Models (YOLO-based)**
- **Player Detection Model** (`hockey-player-detection.pt`): 2-class model detecting players and goalies
- **Puck Detection Model** (`hockey-puck-detection.pt`): Specialized model for puck detection with inference slicing
- **Keypoint Detection Model** (`hockey-detection.pt`): Detects 56 rink keypoints for homography calculation

#### 2. **Core Processing Pipeline**

```
Video Input → Frame Extraction → Detection → Tracking → Classification → Visualization → Output
                                     ↓           ↓           ↓
                                  YOLO      ByteTrack   Team Classifier
                                                              ↓
                                                      Homography Transform
                                                              ↓
                                                        2D Rink Map
```

#### 3. **Key Modules**

##### Detection & Tracking
- `main.py`: Main processing pipeline orchestrating all components
- `common/puck.py`: PuckTracker maintains detection history and smooths trajectories
- `common/smooth_annotator.py`: Provides stable visualization by smoothing annotations

##### Team Classification
- `common/team.py`: Base team classifier using embeddings
- `common/team_hybrid.py`: Advanced hybrid classifier combining:
  - MobileNetV3 deep features
  - Color histogram analysis (HSV/LAB)
  - Spectral clustering
  - Temporal consistency
- `common/team_selector.py`: Interactive UI for manual team selection

##### Position Mapping & Calibration
- `common/rink_keypoint_detector.py`: Detects rink keypoints for homography
- `annotators/rink_annotator.py`: Handles 2D rink visualization
- `common/interactive_calibrator.py`: Interactive calibration system with:
  - Manual keypoint adjustment
  - Camera movement detection
  - Segment-based calibration
  - Quality validation
- `common/homography_stabilizer.py`: Stabilizes homography and player positions

##### Visualization
- `common/styled_label_annotator.py`: Custom label styling
- `configs/hockey.py`: NHL rink dimensions and configuration

### Technical Implementation Details

#### Homography Calculation

The system uses homography transformation to map player positions from video coordinates to 2D rink coordinates:

1. **Keypoint Detection**: 56 rink keypoints are detected using a pose estimation model
2. **Keypoint Filtering**: Only stable keypoints (IDs: 4, 5, 11, 12, 17, 18, 14, 20, 22, 24, 25, 26, 27, 37, 45, 38, 44, 50, 51, 55, 54, 41, 40) are used
3. **Correspondence Mapping**: Keypoints are mapped to known rink positions via `keypoints.json`
4. **RANSAC Homography**: Robust homography calculation with outlier rejection
5. **Validation**: Homography quality checked before application
6. **Stabilization**: Temporal smoothing prevents jumping between frames

#### Team Classification Pipeline

1. **Feature Extraction**:
   - Deep features from pre-trained MobileNetV3
   - Color histograms in HSV and LAB color spaces
   - Saturation and white pixel analysis

2. **Clustering**:
   - Spectral clustering for robust team separation
   - Temporal consistency using tracking history
   - Automatic fallback to white vs. colored detection

3. **Jersey Number Recognition**:
   - EasyOCR for number detection
   - Persistence across frames using tracker IDs
   - Confidence-based filtering

#### Interactive Calibration System

The calibration system provides full control over the homography mapping:

1. **Calibration Modes**:
   - Automatic from detected keypoints
   - Manual keypoint adjustment via drag-and-drop
   - Segment-based for different camera angles

2. **Quality Metrics**:
   - Reprojection error calculation
   - Inlier/outlier ratio
   - Visual feedback on calibration quality

3. **Persistence**:
   - Save/load calibration profiles
   - Per-video calibration storage
   - Automatic calibration reuse

## 🚀 Usage

### Basic Usage

```bash
# Player detection with team classification
python hockey/main.py --source_path video.mp4 --target_path output.mp4 --device cuda

# With 2D rink visualization
python hockey/main.py --source_path video.mp4 --target_path output.mp4 --mode TEAM_CLASSIFICATION --show-2d-map

# Interactive calibration mode
python hockey/main.py --source_path video.mp4 --calibration-mode
```

### Processing Modes

1. **PLAYER_DETECTION**: Detect players/goalies with jersey numbers
2. **PUCK_DETECTION**: Track puck with visual trail
3. **PLAYER_TRACKING**: Track players with persistent IDs
4. **TEAM_CLASSIFICATION**: Classify teams and apply color coding (with optional 2D map)

### Calibration Controls

When in calibration mode:
- **H**: Toggle help display
- **L**: Lock/unlock homography
- **R**: Recalculate homography
- **G**: Save good calibration segment
- **C**: Toggle confidence display
- **S**: Save calibration to file
- **Left click**: Select/drag keypoint
- **Right click**: Remove manual keypoint

## 📁 Project Structure

```
hockey-vision-analytics/
├── hockey/
│   ├── main.py                    # Main entry point
│   ├── configs/
│   │   └── hockey.py             # Rink configuration
│   ├── common/
│   │   ├── team.py              # Team classification
│   │   ├── team_hybrid.py       # Advanced classifier
│   │   ├── puck.py              # Puck tracking
│   │   ├── rink_keypoint_detector.py
│   │   ├── interactive_calibrator.py
│   │   ├── homography_stabilizer.py
│   │   └── ...
│   ├── annotators/
│   │   └── rink_annotator.py    # 2D visualization
│   ├── data/
│   │   └── keypoints.json       # Keypoint mappings
│   └── models/                   # YOLO models
├── notebooks/                    # Training notebooks
└── videos/                       # Sample videos
```

## 🔧 Dependencies

- ultralytics (YOLO)
- supervision (tracking, annotation)
- opencv-python
- numpy
- torch
- scikit-learn
- scipy
- easyocr
- torchvision

## 📊 Performance Considerations

- **GPU Acceleration**: Strongly recommended for real-time processing
- **Model Optimization**: Uses MobileNetV3 for efficient feature extraction
- **Batch Processing**: Processes frames sequentially with tracking persistence
- **Memory Management**: Efficient frame buffering and cleanup

## 🎯 Future Enhancements

- [ ] Multi-camera synchronization
- [ ] Player statistics extraction
- [ ] Real-time streaming support
- [ ] Cloud deployment options
- [ ] Advanced analytics dashboard
- [ ] Training pipeline for custom models

## 📝 Notes

- The system requires initial calibration for accurate 2D mapping
- Team classification works best with clear jersey color differences
- Keypoint detection quality depends on camera angle and video quality
- Homography calculation requires at least 4 high-confidence keypoints

## 🤝 Contributing

Contributions are welcome! Please ensure code quality and add appropriate tests for new features.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.