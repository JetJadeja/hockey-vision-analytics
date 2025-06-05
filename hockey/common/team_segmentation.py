from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict
import supervision as sv


class SegmentationTeamClassifier:
    """
    Team classifier using player segmentation to isolate jersey regions from ice/background.
    Uses semantic segmentation to extract clean jersey pixels for accurate color-based classification.
    """
    
    def __init__(self, device: str = 'cpu', visualize_segmentation: bool = False):
        self.device = device
        self.visualize_segmentation = visualize_segmentation
        
        # Team tracking
        self.player_history: Dict[int, List[int]] = defaultdict(list)
        self.history_window = 10
        
        # Clustering
        self.kmeans = None
        self.team_colors = None
        
        # Store segmentation masks for visualization
        self.last_masks = {}
        
    def segment_player(self, crop: np.ndarray) -> np.ndarray:
        """
        Segment player using GrabCut algorithm for better body isolation.
        Returns binary mask where True = jersey pixels.
        """
        height, width = crop.shape[:2]
        
        # Initialize mask for GrabCut
        mask = np.zeros((height, width), np.uint8)
        
        # Define rectangle around player (leaving margin for background)
        margin_x = int(width * 0.15)
        margin_y = int(height * 0.1)
        rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
        
        # Initialize foreground and background models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Apply GrabCut
            cv2.grabCut(crop, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Extract foreground
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Focus on jersey area (upper body)
            jersey_mask = mask2.copy()
            
            # Remove lower 40% (legs/skates)
            lower_cutoff = int(height * 0.6)
            jersey_mask[lower_cutoff:, :] = 0
            
            # Remove upper 15% (helmet/head) 
            upper_cutoff = int(height * 0.15)
            jersey_mask[:upper_cutoff, :] = 0
            
            # Only keep central vertical strip (avoid arms/sticks)
            left_margin = int(width * 0.25)
            right_margin = int(width * 0.75)
            jersey_mask[:, :left_margin] = 0
            jersey_mask[:, right_margin:] = 0
            
            # Clean up with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            jersey_mask = cv2.morphologyEx(jersey_mask, cv2.MORPH_CLOSE, kernel)
            jersey_mask = cv2.morphologyEx(jersey_mask, cv2.MORPH_OPEN, kernel)
            
            # Find largest connected component (main body)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(jersey_mask, connectivity=8)
            if num_labels > 1:
                # Get largest component (excluding background)
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                jersey_mask = (labels == largest_label).astype(np.uint8)
            
            return jersey_mask.astype(bool)
            
        except Exception as e:
            # Fallback to simple center rectangle
            fallback_mask = np.zeros((height, width), dtype=bool)
            top = int(height * 0.2)
            bottom = int(height * 0.6)
            left = int(width * 0.3)
            right = int(width * 0.7)
            fallback_mask[top:bottom, left:right] = True
            return fallback_mask
    
    def extract_jersey_colors(self, crop: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract color features from segmented jersey region.
        """
        # Apply mask to get only jersey pixels
        jersey_pixels = crop[mask]
        
        if len(jersey_pixels) < 100:  # Not enough pixels
            return {
                'is_white': 0.5,
                'dominant_hue': 0,
                'saturation': 0,
                'brightness': 128
            }
        
        # Convert to different color spaces
        hsv_pixels = cv2.cvtColor(jersey_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        lab_pixels = cv2.cvtColor(jersey_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        
        # White detection in LAB space (more reliable than HSV)
        l_channel = lab_pixels[:, 0]
        a_channel = lab_pixels[:, 1]
        b_channel = lab_pixels[:, 2]
        
        # White pixels have high L and low a,b variance
        white_mask = (l_channel > 200) & (np.abs(a_channel - 128) < 10) & (np.abs(b_channel - 128) < 10)
        white_ratio = np.sum(white_mask) / len(white_mask)
        
        # Get dominant color from non-white pixels
        colored_pixels = hsv_pixels[~white_mask]
        if len(colored_pixels) > 50:
            # Use hue histogram for colored pixels
            hues = colored_pixels[:, 0]
            hist, _ = np.histogram(hues, bins=18, range=(0, 180))
            dominant_hue = np.argmax(hist) * 10  # Convert bin to hue value
            avg_saturation = np.mean(colored_pixels[:, 1])
        else:
            dominant_hue = 0
            avg_saturation = np.mean(hsv_pixels[:, 1])
        
        avg_brightness = np.mean(hsv_pixels[:, 2])
        
        return {
            'is_white': white_ratio,
            'dominant_hue': dominant_hue,
            'saturation': avg_saturation,
            'brightness': avg_brightness
        }
    
    def classify_single_jersey(self, crop: np.ndarray) -> Tuple[int, float]:
        """
        Classify a single jersey using segmentation.
        Returns (team_id, confidence).
        """
        # Segment player
        mask = self.segment_player(crop)
        
        # Extract color features
        features = self.extract_jersey_colors(crop, mask)
        
        # Simple classification based on white ratio
        if features['is_white'] > 0.4:
            # White jersey (away team)
            return 0, features['is_white']
        else:
            # Colored jersey (home team)
            # Confidence based on saturation (more colorful = more confident)
            confidence = min(features['saturation'] / 150, 1.0)
            return 1, confidence
    
    def fit(self, crops: List[np.ndarray], positions: Optional[List[tuple]] = None, 
            frame: Optional[np.ndarray] = None, detections: Optional[sv.Detections] = None) -> None:
        """
        Fit the classifier by analyzing jersey colors after segmentation.
        """
        if len(crops) < 10:
            print("Warning: Very few crops for fitting. Results may be unreliable.")
        
        print(f"Segmenting and analyzing {len(crops)} player crops...")
        
        # Extract features for all crops
        all_features = []
        valid_indices = []
        
        for i, crop in enumerate(crops[:50]):  # Limit for speed during fitting
            mask = self.segment_player(crop)
            
            # Only use crops with good segmentation
            if np.sum(mask) > 500:  # At least 500 jersey pixels
                features = self.extract_jersey_colors(crop, mask)
                all_features.append([
                    features['is_white'],
                    features['dominant_hue'],
                    features['saturation'],
                    features['brightness']
                ])
                valid_indices.append(i)
        
        if len(all_features) < 2:
            print("Not enough valid segmentations. Falling back to simple white detection.")
            return
        
        all_features = np.array(all_features)
        
        # Cluster into 2 teams
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        labels = self.kmeans.fit_predict(all_features)
        
        # Determine which cluster is white team
        cluster_white_ratios = []
        for cluster_id in range(2):
            cluster_mask = labels == cluster_id
            if np.any(cluster_mask):
                avg_white = np.mean(all_features[cluster_mask, 0])
                cluster_white_ratios.append(avg_white)
            else:
                cluster_white_ratios.append(0)
        
        # Assign cluster with higher white ratio as team 0 (away/white)
        if cluster_white_ratios[1] > cluster_white_ratios[0]:
            # Swap labels
            labels = 1 - labels
            self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_[[1, 0]]
        
        print(f"Team 0 (Away/White): avg white ratio = {cluster_white_ratios[0]:.2f}")
        print(f"Team 1 (Home/Colored): avg white ratio = {cluster_white_ratios[1]:.2f}")
        
        # Store team color profiles
        self.team_colors = {
            0: {'is_white': cluster_white_ratios[0], 'name': 'Away (White)'},
            1: {'is_white': cluster_white_ratios[1], 'name': 'Home (Colored)'}
        }
    
    def predict(self, crops: List[np.ndarray], tracker_ids: Optional[np.ndarray] = None, 
                positions: Optional[List[tuple]] = None) -> np.ndarray:
        """
        Predict team assignments using segmentation.
        """
        if not crops:
            return np.array([])
        
        predictions = []
        
        # Clear old masks if visualization is off
        if not self.visualize_segmentation:
            self.last_masks.clear()
        
        for i, crop in enumerate(crops):
            # Use segmentation-based classification
            if self.kmeans is not None:
                # Full feature extraction and clustering
                mask = self.segment_player(crop)
                
                # Store mask for visualization if enabled
                if self.visualize_segmentation and tracker_ids is not None and i < len(tracker_ids):
                    tracker_id = tracker_ids[i]
                    if tracker_id is not None:
                        self.last_masks[int(tracker_id)] = mask
                
                features = self.extract_jersey_colors(crop, mask)
                feature_vec = np.array([[
                    features['is_white'],
                    features['dominant_hue'],
                    features['saturation'],
                    features['brightness']
                ]])
                team = self.kmeans.predict(feature_vec)[0]
            else:
                # Fallback to simple classification
                team, _ = self.classify_single_jersey(crop)
            
            # Apply temporal consistency if available
            if tracker_ids is not None and i < len(tracker_ids):
                tracker_id = tracker_ids[i]
                if tracker_id is not None:
                    tracker_id = int(tracker_id)
                    
                    # Add to history
                    self.player_history[tracker_id].append(team)
                    
                    # Keep only recent history
                    if len(self.player_history[tracker_id]) > self.history_window:
                        self.player_history[tracker_id] = self.player_history[tracker_id][-self.history_window:]
                    
                    # Use majority vote if we have enough history
                    if len(self.player_history[tracker_id]) >= 3:
                        team_counts = np.bincount(self.player_history[tracker_id])
                        team = np.argmax(team_counts)
            
            predictions.append(team)
        
        return np.array(predictions)
    
    def get_segmentation_masks(self, tracker_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Get stored segmentation masks for visualization.
        """
        result = {}
        for tid in tracker_ids:
            if tid in self.last_masks:
                result[tid] = self.last_masks[tid]
        return result