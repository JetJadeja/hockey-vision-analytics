from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from collections import defaultdict

# Import the hybrid classifier
try:
    from .team_hybrid import HybridTeamClassifier
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Import the robust classifier
try:
    from .team_robust import RobustTeamClassifier
    ROBUST_AVAILABLE = True
except ImportError:
    ROBUST_AVAILABLE = False



class TeamClassifier:
    """
    Simple hockey team classifier based on white (away) vs colored (home) jerseys.
    Uses basic color analysis to distinguish teams without complex ML.
    """
    
    def __init__(self, device: str = 'cpu', batch_size: int = 32, use_hybrid: bool = True, use_robust: bool = True):
        # Keep interface compatible but ignore device/batch_size for this simple approach
        self.device = device
        self.batch_size = batch_size
        self.use_robust = use_robust and ROBUST_AVAILABLE
        self.use_hybrid = use_hybrid and HYBRID_AVAILABLE and not self.use_robust
        
        if self.use_robust:
            # Use state-of-the-art robust classifier
            self.robust_classifier = RobustTeamClassifier(device=device)
        elif self.use_hybrid:
            # Use advanced hybrid classifier
            self.hybrid_classifier = HybridTeamClassifier(device=device)
        else:
            # Fallback to simple approach
            # Temporal consistency tracking
            self.player_history: Dict[int, List[int]] = defaultdict(list)
            self.history_window = 10  # frames to consider for temporal consistency
            
            # Team assignments (0=away/white, 1=home/colored)
            self.team_assignments = {0: "away", 1: "home"}
    
    def extract_jersey_region(self, crop: np.ndarray) -> np.ndarray:
        """Extract the central jersey area from a player crop."""
        height, width = crop.shape[:2]
        
        # Skip very small crops
        if height < 30 or width < 20:
            return crop
        
        # Focus on torso area
        # Vertical: middle 50% (skip head and legs)
        top = int(height * 0.25)
        bottom = int(height * 0.75)
        
        # Horizontal: center 40% (avoid arms and edges)
        left = int(width * 0.3)
        right = int(width * 0.7)
        
        jersey_region = crop[top:bottom, left:right]
        
        # Ensure we got a valid region
        if jersey_region.size == 0:
            return crop
        
        return jersey_region
    
    def classify_jersey(self, crop: np.ndarray) -> tuple[int, float]:
        """
        Classify a single jersey as white (0) or colored (1).
        Returns (team_id, confidence).
        """
        # Extract jersey region
        jersey = self.extract_jersey_region(crop)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        
        # Calculate color metrics
        avg_brightness = np.mean(hsv[:, :, 2])  # V channel
        avg_saturation = np.mean(hsv[:, :, 1])  # S channel
        
        # Count white pixels (high brightness, low saturation)
        white_mask = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30)
        white_pixel_ratio = np.sum(white_mask) / white_mask.size
        
        # Binary classification with confidence
        if white_pixel_ratio > 0.3 or (avg_brightness > 180 and avg_saturation < 50):
            # White jersey (away team)
            team = 0
            # Confidence based on how "white" it is
            confidence = min(white_pixel_ratio * 2, 1.0)
        else:
            # Colored jersey (home team)
            team = 1
            # Confidence based on saturation (more colorful = more confident)
            confidence = min(avg_saturation / 150, 1.0)
        
        return team, confidence
    
    def fit(self, crops: List[np.ndarray], positions: Optional[List[tuple]] = None) -> None:
        """
        Fit the classifier on training crops.
        Uses robust approach if available, then hybrid, otherwise simple analysis.
        """
        if self.use_robust:
            # Use robust classifier
            try:
                self.robust_classifier.fit(crops, positions=positions)
            except Exception as e:
                print(f"Robust classifier failed: {e}")
                print("Falling back to hybrid classifier")
                self.use_robust = False
                self.use_hybrid = HYBRID_AVAILABLE
                if self.use_hybrid:
                    self.hybrid_classifier = HybridTeamClassifier(device=self.device)
                    self.fit(crops, positions)
                else:
                    self._simple_fit(crops)
        elif self.use_hybrid:
            # Use hybrid classifier
            try:
                self.hybrid_classifier.fit(crops)
            except Exception as e:
                print(f"Hybrid classifier failed: {e}")
                print("Falling back to simple classifier")
                self.use_hybrid = False
                self._simple_fit(crops)
        else:
            self._simple_fit(crops)
    
    def _simple_fit(self, crops: List[np.ndarray]) -> None:
        """Simple fit method for fallback."""
        # Analyze the distribution to verify our assumptions
        white_count = 0
        colored_count = 0
        
        for crop in crops[:100]:  # Sample first 100 for speed
            team, _ = self.classify_jersey(crop)
            if team == 0:
                white_count += 1
            else:
                colored_count += 1
        
        print(f"Sample distribution - White jerseys: {white_count}, Colored jerseys: {colored_count}")
        
        # Could adjust thresholds here if needed, but keeping it simple
    
    def predict(self, crops: List[np.ndarray], tracker_ids: Optional[np.ndarray] = None, positions: Optional[List[tuple]] = None) -> np.ndarray:
        """
        Predict team assignments for player crops.
        If tracker_ids provided, uses temporal consistency.
        """
        if not crops:
            return np.array([])
        
        if self.use_robust:
            # Use robust classifier
            try:
                assignments = self.robust_classifier.predict(crops, tracker_ids, positions)
                return self.robust_classifier.get_team_labels(assignments)
            except Exception as e:
                print(f"Robust prediction failed: {e}")
                print("Falling back to hybrid classifier")
                self.use_robust = False
                self.use_hybrid = HYBRID_AVAILABLE
                if self.use_hybrid:
                    self.hybrid_classifier = HybridTeamClassifier(device=self.device)
        
        if self.use_hybrid:
            # Use hybrid classifier
            try:
                return self.hybrid_classifier.predict(crops, tracker_ids)
            except Exception as e:
                print(f"Hybrid prediction failed: {e}")
                print("Falling back to simple classifier")
                self.use_hybrid = False
        
        # Simple classification fallback
        predictions = []
        
        for i, crop in enumerate(crops):
            # Get initial classification
            team, confidence = self.classify_jersey(crop)
            
            # Apply temporal consistency if we have tracking
            if tracker_ids is not None and i < len(tracker_ids):
                tracker_id = tracker_ids[i]
                if tracker_id is not None:
                    # Convert to int for dictionary key
                    tracker_id = int(tracker_id)
                    
                    # Add to history
                    self.player_history[tracker_id].append(team)
                    
                    # Keep only recent history
                    if len(self.player_history[tracker_id]) > self.history_window:
                        self.player_history[tracker_id] = self.player_history[tracker_id][-self.history_window:]
                    
                    # Use majority vote if we have enough history
                    if len(self.player_history[tracker_id]) >= 3:
                        # Get most common team assignment
                        team_counts = np.bincount(self.player_history[tracker_id])
                        team = np.argmax(team_counts)
            
            predictions.append(team)
        
        return np.array(predictions)