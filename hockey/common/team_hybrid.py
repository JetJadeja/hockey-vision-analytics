from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import supervision as sv


class HybridTeamClassifier:
    """
    Advanced team classifier using deep features, color analysis, and spatial-temporal consistency.
    Combines multiple signals for robust team detection in hockey broadcasts.
    """
    
    def __init__(self, device: str = 'cpu', n_clusters: int = 2):
        self.device = device
        self.n_clusters = n_clusters
        
        # Initialize MobileNetV3 for deep feature extraction
        self.feature_extractor = models.mobilenet_v3_small(pretrained=True)
        # Remove classifier, keep only features
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        
        # Image preprocessing for MobileNet
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Consistent size for jersey regions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Feature scaler for normalization
        self.scaler = StandardScaler()
        
        # Temporal tracking
        self.player_history: Dict[int, List[int]] = defaultdict(list)
        self.history_window = 15
        
        # Clustering model
        self.clusterer = None
        self.cluster_labels = None
        
    def extract_jersey_region(self, crop: np.ndarray) -> np.ndarray:
        """Extract jersey region focusing on upper body."""
        height, width = crop.shape[:2]
        
        if height < 40 or width < 20:
            return crop
            
        # Focus on upper 60% (more than before to get full jersey)
        top = int(height * 0.1)
        bottom = int(height * 0.6)
        
        # Center 60% horizontally
        left = int(width * 0.2)
        right = int(width * 0.8)
        
        return crop[top:bottom, left:right]
    
    def extract_deep_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract deep features using MobileNet."""
        features = []
        
        with torch.no_grad():
            for crop in crops:
                # Extract jersey region
                jersey = self.extract_jersey_region(crop)
                
                # Preprocess for MobileNet
                try:
                    img_tensor = self.preprocess(jersey).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    feat = self.feature_extractor(img_tensor)
                    feat = feat.squeeze().cpu().numpy()
                    features.append(feat)
                except:
                    # If preprocessing fails, use zeros
                    features.append(np.zeros(576))  # MobileNetV3 small output size
                    
        return np.array(features)
    
    def extract_color_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract robust color features."""
        features = []
        
        for crop in crops:
            jersey = self.extract_jersey_region(crop)
            
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)
            
            # HSV features
            h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            
            # Normalize histograms
            h_hist = h_hist / (h_hist.sum() + 1e-7)
            s_hist = s_hist / (s_hist.sum() + 1e-7)
            v_hist = v_hist / (v_hist.sum() + 1e-7)
            
            # Color statistics
            hsv_mean = hsv.mean(axis=(0, 1))
            hsv_std = hsv.std(axis=(0, 1))
            
            # LAB statistics (good for perceptual differences)
            lab_mean = lab.mean(axis=(0, 1))
            lab_std = lab.std(axis=(0, 1))
            
            # Saturation ratio (key for white detection)
            low_sat_ratio = np.sum(hsv[:, :, 1] < 30) / hsv[:, :, 1].size
            high_sat_ratio = np.sum(hsv[:, :, 1] > 100) / hsv[:, :, 1].size
            
            # White pixel ratio
            white_pixels = np.sum((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30))
            white_ratio = white_pixels / hsv[:, :, 1].size
            
            # Combine all color features
            color_feat = np.concatenate([
                h_hist,                    # 18 dims
                s_hist,                    # 8 dims
                v_hist,                    # 8 dims
                hsv_mean / 255,           # 3 dims
                hsv_std / 255,            # 3 dims
                lab_mean / 255,           # 3 dims
                lab_std / 255,            # 3 dims
                [low_sat_ratio],          # 1 dim
                [high_sat_ratio],         # 1 dim
                [white_ratio]             # 1 dim
            ])  # Total: 49 dims
            
            features.append(color_feat)
            
        return np.array(features)
    
    def extract_all_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Combine deep and color features."""
        # Get both feature types
        deep_features = self.extract_deep_features(crops)
        color_features = self.extract_color_features(crops)
        
        # Concatenate features
        combined_features = np.hstack([deep_features, color_features])
        
        return combined_features
    
    def fit(self, crops: List[np.ndarray], positions: Optional[List[Tuple[float, float]]] = None) -> None:
        """Fit the classifier on training crops."""
        if len(crops) < self.n_clusters * 2:
            raise ValueError(f"Need at least {self.n_clusters * 2} crops for clustering")
        
        print(f"Extracting features from {len(crops)} crops...")
        
        # Extract all features
        features = self.extract_all_features(crops)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Add spatial features if positions provided
        if positions and len(positions) == len(crops):
            # Normalize positions to [0, 1]
            positions_array = np.array(positions)
            pos_min = positions_array.min(axis=0)
            pos_max = positions_array.max(axis=0)
            positions_normalized = (positions_array - pos_min) / (pos_max - pos_min + 1e-7)
            
            # Add position features with lower weight
            features_normalized = np.hstack([
                features_normalized,
                positions_normalized * 0.1  # Lower weight for positions
            ])
        
        print("Performing spectral clustering...")
        
        # Spectral clustering with RBF kernel
        self.clusterer = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='rbf',
            gamma=1.0,
            n_init=10,
            random_state=42
        )
        
        self.cluster_labels = self.clusterer.fit_predict(features_normalized)
        
        # Analyze clusters to determine which is white/colored
        self._analyze_clusters(crops, self.cluster_labels)
        
    def _analyze_clusters(self, crops: List[np.ndarray], labels: np.ndarray) -> None:
        """Analyze clusters to identify white vs colored teams."""
        cluster_stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_crops = [crop for crop, label in zip(crops, labels) if label == cluster_id]
            
            if cluster_crops:
                # Calculate average saturation for cluster
                saturations = []
                white_ratios = []
                
                for crop in cluster_crops[:20]:  # Sample for speed
                    jersey = self.extract_jersey_region(crop)
                    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
                    
                    avg_saturation = np.mean(hsv[:, :, 1])
                    saturations.append(avg_saturation)
                    
                    white_pixels = np.sum((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30))
                    white_ratio = white_pixels / hsv[:, :, 1].size
                    white_ratios.append(white_ratio)
                
                cluster_stats[cluster_id] = {
                    'avg_saturation': np.mean(saturations),
                    'avg_white_ratio': np.mean(white_ratios),
                    'count': len(cluster_crops)
                }
        
        # Identify white team (lowest saturation)
        if len(cluster_stats) == 2:
            white_cluster = min(cluster_stats.keys(), 
                              key=lambda k: cluster_stats[k]['avg_saturation'])
            
            # Swap labels if needed so 0=white, 1=colored
            if white_cluster == 1:
                self.cluster_labels = 1 - self.cluster_labels
                
            print(f"Cluster 0 (White/Away): {cluster_stats[0]['count']} players, "
                  f"avg saturation: {cluster_stats[0]['avg_saturation']:.1f}")
            print(f"Cluster 1 (Colored/Home): {cluster_stats[1]['count']} players, "
                  f"avg saturation: {cluster_stats[1]['avg_saturation']:.1f}")
    
    def predict(self, crops: List[np.ndarray], tracker_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict team assignments for new crops."""
        if not crops:
            return np.array([])
        
        if self.clusterer is None:
            # Fallback to simple classification if not fitted
            return self._simple_classify(crops, tracker_ids)
        
        # Extract features
        features = self.extract_all_features(crops)
        features_normalized = self.scaler.transform(features)
        
        # For spectral clustering, we need to use the training data
        # So we'll use a different approach: find nearest cluster center
        predictions = self._predict_by_similarity(features_normalized)
        
        # Apply temporal consistency
        if tracker_ids is not None:
            predictions = self._apply_temporal_consistency(predictions, tracker_ids)
        
        return predictions
    
    def _predict_by_similarity(self, features: np.ndarray) -> np.ndarray:
        """Predict by finding most similar training samples."""
        # For now, use simple white detection as fallback
        # In production, would store training features and use kNN
        predictions = []
        
        for feat in features:
            # Use white ratio from color features (last element)
            white_ratio = feat[-1]
            saturation_features = feat[-10:-7]  # Saturation histogram
            
            if white_ratio > 0.3 or np.argmax(saturation_features) == 0:
                predictions.append(0)  # White team
            else:
                predictions.append(1)  # Colored team
                
        return np.array(predictions)
    
    def _simple_classify(self, crops: List[np.ndarray], tracker_ids: Optional[np.ndarray]) -> np.ndarray:
        """Simple fallback classification."""
        predictions = []
        
        for crop in crops:
            jersey = self.extract_jersey_region(crop)
            hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
            
            # Simple white detection
            avg_saturation = np.mean(hsv[:, :, 1])
            white_pixels = np.sum((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30))
            white_ratio = white_pixels / hsv[:, :, 1].size
            
            if white_ratio > 0.25 or avg_saturation < 40:
                predictions.append(0)  # White
            else:
                predictions.append(1)  # Colored
        
        predictions = np.array(predictions)
        
        # Apply temporal consistency
        if tracker_ids is not None:
            predictions = self._apply_temporal_consistency(predictions, tracker_ids)
            
        return predictions
    
    def _apply_temporal_consistency(self, predictions: np.ndarray, tracker_ids: np.ndarray) -> np.ndarray:
        """Apply temporal consistency using tracking history."""
        smoothed_predictions = predictions.copy()
        
        for i, (pred, tracker_id) in enumerate(zip(predictions, tracker_ids)):
            if tracker_id is not None:
                tracker_id = int(tracker_id)
                
                # Add to history
                self.player_history[tracker_id].append(pred)
                
                # Keep window size
                if len(self.player_history[tracker_id]) > self.history_window:
                    self.player_history[tracker_id] = self.player_history[tracker_id][-self.history_window:]
                
                # Use majority vote if enough history
                if len(self.player_history[tracker_id]) >= 5:
                    team_counts = np.bincount(self.player_history[tracker_id])
                    smoothed_predictions[i] = np.argmax(team_counts)
                    
        return smoothed_predictions