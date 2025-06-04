from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np
import cv2
import torch
from transformers import AutoModel, AutoProcessor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from collections import defaultdict, Counter
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TeamAssignment:
    """Represents a team assignment with confidence."""
    team_id: int
    confidence: float
    is_outlier: bool = False


@dataclass
class PlayerProfile:
    """Maintains temporal information about a player."""
    tracker_id: int
    team_history: List[int]
    confidence_history: List[float]
    embedding_history: List[np.ndarray]
    last_seen_frame: int
    
    def get_stable_team(self, min_confidence: float = 0.7) -> Optional[int]:
        """Get most stable team assignment."""
        if not self.team_history:
            return None
        
        # Filter by confidence
        confident_teams = [
            team for team, conf in zip(self.team_history, self.confidence_history)
            if conf >= min_confidence
        ]
        
        if not confident_teams:
            return Counter(self.team_history).most_common(1)[0][0]
        
        return Counter(confident_teams).most_common(1)[0][0]


class RobustTeamClassifier:
    """
    State-of-the-art team classification for hockey using:
    - SigLIP embeddings with optimized preprocessing
    - HDBSCAN for robust clustering
    - Multi-modal features (visual + color + spatial)
    - Temporal consistency with confidence weighting
    - Continuous refinement
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 model_name: str = 'google/siglip-base-patch16-256',
                 embedding_dim: int = 768,
                 min_cluster_size: int = 5,
                 min_samples: int = 3):
        
        self.device = device
        
        # Load SigLIP model
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Feature processing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50, random_state=42)
        
        # Clustering
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.clusterer = None
        
        # Team mapping
        self.team_mapping = {}  # cluster_id -> team_id
        self.team_profiles = {}  # team_id -> team statistics
        
        # Player tracking
        self.player_profiles: Dict[int, PlayerProfile] = {}
        self.current_frame = 0
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.4
        
    def preprocess_crop(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intelligent preprocessing to focus on jersey regions.
        Returns: (processed_crop, jersey_mask)
        """
        height, width = crop.shape[:2]
        
        # Create jersey-focused mask
        jersey_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Detect and mask out jersey numbers (bright regions in center)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Find bright regions (potential numbers)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Focus on center region where numbers typically are
        center_x, center_y = width // 2, height // 2
        number_region = np.zeros_like(bright_mask)
        cv2.ellipse(number_region, (center_x, int(center_y * 0.8)), 
                   (int(width * 0.3), int(height * 0.2)), 0, 0, 360, 255, -1)
        
        # Combine to find number regions
        number_mask = cv2.bitwise_and(bright_mask, number_region)
        
        # Dilate to cover full number area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        number_mask = cv2.morphologyEx(number_mask, cv2.MORPH_DILATE, kernel)
        
        # Update jersey mask (exclude numbers)
        jersey_mask = cv2.bitwise_not(number_mask)
        
        # Apply mild Gaussian blur to reduce texture details
        processed = cv2.GaussianBlur(crop, (3, 3), 0)
        
        # Enhance jersey regions
        jersey_region = cv2.bitwise_and(processed, processed, mask=jersey_mask)
        
        return jersey_region, jersey_mask
    
    def extract_siglip_features(self, crops: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extract SigLIP features with jersey-focused preprocessing."""
        features = []
        
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            
            # Preprocess crops
            processed_crops = []
            for crop in batch_crops:
                processed, _ = self.preprocess_crop(crop)
                processed_crops.append(processed)
            
            # Process through SigLIP
            inputs = self.processor(images=processed_crops, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                batch_features = outputs.cpu().numpy()
            
            features.append(batch_features)
        
        return np.vstack(features) if features else np.array([])
    
    def extract_color_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract color features with jersey mask applied."""
        features = []
        
        for crop in crops:
            _, jersey_mask = self.preprocess_crop(crop)
            
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            
            # Apply jersey mask
            hsv_masked = cv2.bitwise_and(hsv, hsv, mask=jersey_mask)
            lab_masked = cv2.bitwise_and(lab, lab, mask=jersey_mask)
            
            # Compute masked histograms
            h_hist = cv2.calcHist([hsv_masked], [0], jersey_mask, [18], [0, 180])
            s_hist = cv2.calcHist([hsv_masked], [1], jersey_mask, [16], [0, 256])
            
            # Normalize
            h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-7)
            s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-7)
            
            # Color statistics on masked regions
            mask_bool = jersey_mask > 0
            if np.any(mask_bool):
                hsv_mean = hsv_masked[mask_bool].mean(axis=0)
                lab_mean = lab_masked[mask_bool].mean(axis=0)
                
                # Saturation analysis
                saturations = hsv_masked[:, :, 1][mask_bool]
                low_sat_ratio = np.sum(saturations < 30) / len(saturations)
                med_sat_ratio = np.sum((saturations >= 30) & (saturations < 100)) / len(saturations)
                high_sat_ratio = np.sum(saturations >= 100) / len(saturations)
            else:
                hsv_mean = np.zeros(3)
                lab_mean = np.zeros(3)
                low_sat_ratio = med_sat_ratio = high_sat_ratio = 0
            
            # Combine features
            color_feat = np.concatenate([
                h_hist,                          # 18 dims
                s_hist,                          # 16 dims
                hsv_mean / 255,                  # 3 dims
                lab_mean / 255,                  # 3 dims
                [low_sat_ratio, med_sat_ratio, high_sat_ratio]  # 3 dims
            ])  # Total: 43 dims
            
            features.append(color_feat)
        
        return np.array(features)
    
    def extract_multimodal_features(self, 
                                  crops: List[np.ndarray],
                                  positions: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """Extract and combine multiple feature modalities."""
        # Visual features from SigLIP
        visual_features = self.extract_siglip_features(crops)
        
        # Color features
        color_features = self.extract_color_features(crops)
        
        # Combine features
        combined = np.hstack([visual_features, color_features])
        
        # Add spatial features if available
        if positions and len(positions) == len(crops):
            pos_array = np.array(positions)
            # Normalize positions
            pos_normalized = (pos_array - pos_array.mean(axis=0)) / (pos_array.std(axis=0) + 1e-7)
            combined = np.hstack([combined, pos_normalized * 0.1])  # Lower weight
        
        return combined
    
    def fit(self, 
            crops: List[np.ndarray],
            positions: Optional[List[Tuple[float, float]]] = None,
            sample_every_n: int = 1) -> None:
        """
        Fit classifier using HDBSCAN for robust clustering.
        """
        if len(crops) < self.min_cluster_size * 2:
            raise ValueError(f"Need at least {self.min_cluster_size * 2} crops")
        
        # Sample crops if too many
        if sample_every_n > 1:
            indices = list(range(0, len(crops), sample_every_n))
            crops = [crops[i] for i in indices]
            if positions:
                positions = [positions[i] for i in indices]
        
        print(f"Extracting features from {len(crops)} crops...")
        
        # Extract features
        features = self.extract_multimodal_features(crops, positions)
        
        # Normalize and reduce dimensionality
        features_scaled = self.scaler.fit_transform(features)
        features_reduced = self.pca.fit_transform(features_scaled)
        
        print("Performing HDBSCAN clustering...")
        
        # HDBSCAN clustering
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        cluster_labels = self.clusterer.fit_predict(features_reduced)
        
        # Analyze clusters
        self._analyze_and_map_clusters(crops, cluster_labels, features_reduced)
    
    def _analyze_and_map_clusters(self, 
                                 crops: List[np.ndarray], 
                                 labels: np.ndarray,
                                 features: np.ndarray) -> None:
        """Analyze clusters and create team mapping."""
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove outliers
        
        if len(unique_labels) < 2:
            print(f"Warning: Only {len(unique_labels)} clusters found. Using fallback.")
            self._fallback_clustering(crops, labels)
            return
        
        # Analyze each cluster
        cluster_stats = {}
        
        for label in unique_labels:
            mask = labels == label
            cluster_crops = [crop for crop, m in zip(crops, mask) if m]
            
            # Sample crops for analysis
            sample_size = min(20, len(cluster_crops))
            sample_indices = np.random.choice(len(cluster_crops), sample_size, replace=False)
            
            # Compute cluster statistics
            saturations = []
            white_ratios = []
            
            for idx in sample_indices:
                crop = cluster_crops[idx]
                _, jersey_mask = self.preprocess_crop(crop)
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                
                # Masked statistics
                mask_bool = jersey_mask > 0
                if np.any(mask_bool):
                    sat_values = hsv[:, :, 1][mask_bool]
                    saturations.append(np.median(sat_values))
                    
                    # White detection
                    white_pixels = np.sum((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30))
                    white_ratios.append(white_pixels / np.sum(mask_bool))
            
            cluster_stats[label] = {
                'size': np.sum(mask),
                'median_saturation': np.median(saturations),
                'white_ratio': np.median(white_ratios),
                'cohesion': self.clusterer.probabilities_[mask].mean()
            }
        
        # Find two main teams (largest clusters with good cohesion)
        sorted_clusters = sorted(
            cluster_stats.items(),
            key=lambda x: x[1]['size'] * x[1]['cohesion'],
            reverse=True
        )
        
        # Map clusters to teams
        if len(sorted_clusters) >= 2:
            # Determine white vs colored based on saturation
            team_clusters = sorted_clusters[:2]
            
            if team_clusters[0][1]['median_saturation'] < team_clusters[1][1]['median_saturation']:
                self.team_mapping[team_clusters[0][0]] = 0  # White/Away
                self.team_mapping[team_clusters[1][0]] = 1  # Colored/Home
            else:
                self.team_mapping[team_clusters[0][0]] = 1  # Colored/Home
                self.team_mapping[team_clusters[1][0]] = 0  # White/Away
            
            # Store team profiles
            for cluster_id, team_id in self.team_mapping.items():
                self.team_profiles[team_id] = {
                    'cluster_id': cluster_id,
                    'stats': cluster_stats[cluster_id],
                    'exemplar_features': features[labels == cluster_id].mean(axis=0)
                }
        
        # Print summary
        for team_id, profile in self.team_profiles.items():
            stats = profile['stats']
            team_type = "White/Away" if team_id == 0 else "Colored/Home"
            print(f"Team {team_id} ({team_type}): {stats['size']} players, "
                  f"saturation: {stats['median_saturation']:.1f}, "
                  f"cohesion: {stats['cohesion']:.2f}")
    
    def _fallback_clustering(self, crops: List[np.ndarray], labels: np.ndarray) -> None:
        """Fallback to simple white vs colored detection."""
        white_indices = []
        colored_indices = []
        
        for i, crop in enumerate(crops):
            _, jersey_mask = self.preprocess_crop(crop)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            mask_bool = jersey_mask > 0
            if np.any(mask_bool):
                sat_median = np.median(hsv[:, :, 1][mask_bool])
                
                if sat_median < 40:
                    white_indices.append(i)
                    labels[i] = 0
                else:
                    colored_indices.append(i)
                    labels[i] = 1
        
        self.team_mapping = {0: 0, 1: 1}
        print(f"Fallback clustering: {len(white_indices)} white, {len(colored_indices)} colored")
    
    def predict(self, 
               crops: List[np.ndarray],
               tracker_ids: Optional[np.ndarray] = None,
               positions: Optional[List[Tuple[float, float]]] = None) -> List[TeamAssignment]:
        """
        Predict team assignments with confidence scores.
        """
        if not crops:
            return []
        
        self.current_frame += 1
        
        # Extract features
        features = self.extract_multimodal_features(crops, positions)
        features_scaled = self.scaler.transform(features)
        features_reduced = self.pca.transform(features_scaled)
        
        assignments = []
        
        if self.clusterer and hasattr(self.clusterer, 'predict'):
            # Use HDBSCAN soft clustering
            labels, strengths = hdbscan.approximate_predict(self.clusterer, features_reduced)
            
            for i, (label, strength) in enumerate(zip(labels, strengths)):
                # Handle outliers
                if label == -1:
                    # Try to assign based on history or color
                    assignment = self._handle_outlier(crops[i], tracker_ids[i] if tracker_ids is not None else None)
                else:
                    # Map cluster to team
                    team_id = self.team_mapping.get(label, -1)
                    if team_id == -1:
                        assignment = self._handle_outlier(crops[i], tracker_ids[i] if tracker_ids is not None else None)
                    else:
                        assignment = TeamAssignment(team_id, float(strength), False)
                
                # Apply temporal consistency
                if tracker_ids is not None and i < len(tracker_ids) and tracker_ids[i] is not None:
                    assignment = self._apply_temporal_consistency(
                        assignment, int(tracker_ids[i]), features_reduced[i]
                    )
                
                assignments.append(assignment)
        else:
            # Fallback prediction
            for i, crop in enumerate(crops):
                assignment = self._simple_predict(crop)
                
                if tracker_ids is not None and i < len(tracker_ids) and tracker_ids[i] is not None:
                    assignment = self._apply_temporal_consistency(
                        assignment, int(tracker_ids[i]), features_reduced[i]
                    )
                
                assignments.append(assignment)
        
        return assignments
    
    def _handle_outlier(self, crop: np.ndarray, tracker_id: Optional[int]) -> TeamAssignment:
        """Handle outlier detection with fallback methods."""
        # Check tracking history first
        if tracker_id is not None and tracker_id in self.player_profiles:
            profile = self.player_profiles[tracker_id]
            stable_team = profile.get_stable_team()
            if stable_team is not None:
                return TeamAssignment(stable_team, 0.6, True)
        
        # Fallback to simple color detection
        return self._simple_predict(crop)
    
    def _simple_predict(self, crop: np.ndarray) -> TeamAssignment:
        """Simple color-based prediction."""
        _, jersey_mask = self.preprocess_crop(crop)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        mask_bool = jersey_mask > 0
        if np.any(mask_bool):
            sat_values = hsv[:, :, 1][mask_bool]
            sat_median = np.median(sat_values)
            
            if sat_median < 40:
                confidence = 1.0 - (sat_median / 40)
                return TeamAssignment(0, confidence, False)  # White
            else:
                confidence = min(sat_median / 100, 1.0)
                return TeamAssignment(1, confidence, False)  # Colored
        
        return TeamAssignment(0, 0.5, True)  # Default uncertain
    
    def _apply_temporal_consistency(self, 
                                  assignment: TeamAssignment,
                                  tracker_id: int,
                                  embedding: np.ndarray) -> TeamAssignment:
        """Apply temporal smoothing and confidence adjustment."""
        # Update or create player profile
        if tracker_id not in self.player_profiles:
            self.player_profiles[tracker_id] = PlayerProfile(
                tracker_id=tracker_id,
                team_history=[],
                confidence_history=[],
                embedding_history=[],
                last_seen_frame=self.current_frame
            )
        
        profile = self.player_profiles[tracker_id]
        profile.team_history.append(assignment.team_id)
        profile.confidence_history.append(assignment.confidence)
        profile.embedding_history.append(embedding)
        profile.last_seen_frame = self.current_frame
        
        # Limit history
        max_history = 20
        if len(profile.team_history) > max_history:
            profile.team_history = profile.team_history[-max_history:]
            profile.confidence_history = profile.confidence_history[-max_history:]
            profile.embedding_history = profile.embedding_history[-max_history:]
        
        # Get stable team with weighted voting
        stable_team = profile.get_stable_team(min_confidence=0.6)
        
        if stable_team is not None and len(profile.team_history) >= 5:
            # Calculate consistency bonus
            recent_teams = profile.team_history[-5:]
            consistency = recent_teams.count(stable_team) / len(recent_teams)
            
            # Adjust confidence based on consistency
            if stable_team == assignment.team_id:
                new_confidence = min(assignment.confidence + consistency * 0.2, 1.0)
            else:
                # Override if very consistent history disagrees
                if consistency > 0.8:
                    return TeamAssignment(stable_team, consistency, assignment.is_outlier)
                new_confidence = assignment.confidence * (1 - consistency * 0.3)
            
            return TeamAssignment(assignment.team_id, new_confidence, assignment.is_outlier)
        
        return assignment
    
    def get_team_labels(self, assignments: List[TeamAssignment]) -> np.ndarray:
        """Convert TeamAssignment list to simple numpy array of labels."""
        return np.array([a.team_id for a in assignments])
    
    def get_confidences(self, assignments: List[TeamAssignment]) -> np.ndarray:
        """Extract confidence scores."""
        return np.array([a.confidence for a in assignments])