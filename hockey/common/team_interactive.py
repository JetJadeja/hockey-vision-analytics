"""
Interactive Team Classification System
User clicks on one player from each team to establish ground truth
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
import cv2
import supervision as sv
from collections import defaultdict


@dataclass
class TeamExample:
    """Stores user-selected example for a team"""
    team_id: int
    crop: np.ndarray
    features: Dict[str, np.ndarray]
    bbox: np.ndarray
    frame: np.ndarray


@dataclass
class TeamExamples:
    """Stores multiple examples for a team"""
    team_id: int
    examples: List[TeamExample]
    
    def add_example(self, example: TeamExample):
        self.examples.append(example)
    
    def get_all_features(self) -> List[Dict[str, np.ndarray]]:
        return [ex.features for ex in self.examples]


class InteractiveTeamClassifier:
    """
    Semi-supervised team classifier that uses user input to establish ground truth
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.team_examples: Dict[int, TeamExamples] = {}
        self.player_history: Dict[int, List[int]] = defaultdict(list)
        self.confidence_threshold = 0.7
        
        # Click handling
        self.click_position = None
        self.current_selections = []  # Multiple selections
        self.min_examples_per_team = 2
        self.max_examples_per_team = 5
        
    def get_user_team_selections(self, 
                                frame: np.ndarray, 
                                detections: sv.Detections,
                                team_name: str = "Team") -> List[int]:
        """
        Display frame and let user click on multiple players from specified team
        Returns list of indices of selected detections
        """
        # Create display frame with bounding boxes
        display_frame = frame.copy()
        
        # Draw all detections
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, str(i), (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(display_frame, f"Click {self.min_examples_per_team}-{self.max_examples_per_team} players from {team_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press SPACE when done, 'r' to reset, 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Setup mouse callback
        self.click_position = None
        self.current_selections = []
        cv2.namedWindow("Team Selection")
        cv2.setMouseCallback("Team Selection", self._mouse_callback_multi, 
                           param={'detections': detections})
        
        while True:
            # Highlight selections
            display_copy = display_frame.copy()
            
            # Draw all selected players
            for idx in self.current_selections:
                bbox = detections.xyxy[idx]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(display_copy, "SELECTED", (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Show selection count
            cv2.putText(display_copy, f"Selected: {len(self.current_selections)} players", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show if ready to confirm
            if len(self.current_selections) >= self.min_examples_per_team:
                cv2.putText(display_copy, "Press SPACE to confirm selections", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Team Selection", display_copy)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit without selection
                cv2.destroyWindow("Team Selection")
                return []
            elif key == ord('r'):  # Reset selections
                self.current_selections = []
            elif key == ord(' ') and len(self.current_selections) >= self.min_examples_per_team:  # Confirm
                cv2.destroyWindow("Team Selection")
                return self.current_selections
    
    def _mouse_callback_multi(self, event, x, y, flags, param):
        """Handle mouse clicks for multiple player selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            detections = param['detections']
            
            # Find which detection was clicked
            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Toggle selection
                    if i in self.current_selections:
                        self.current_selections.remove(i)
                    elif len(self.current_selections) < self.max_examples_per_team:
                        self.current_selections.append(i)
                    break
    
    def extract_features(self, crop: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple feature types for robust matching
        """
        features = {}
        
        # Color histogram features
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 1. Full histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        features['h_hist'] = h_hist.flatten() / (h_hist.sum() + 1e-7)
        features['s_hist'] = s_hist.flatten() / (s_hist.sum() + 1e-7)
        features['v_hist'] = v_hist.flatten() / (v_hist.sum() + 1e-7)
        
        # 2. Statistical features
        features['hsv_mean'] = np.mean(hsv, axis=(0, 1)) / 255
        features['hsv_std'] = np.std(hsv, axis=(0, 1)) / 255
        
        # 3. Spatial color distribution (divide into quadrants)
        h, w = crop.shape[:2]
        quadrants = [
            crop[0:h//2, 0:w//2],      # top-left
            crop[0:h//2, w//2:w],      # top-right
            crop[h//2:h, 0:w//2],      # bottom-left
            crop[h//2:h, w//2:w]       # bottom-right
        ]
        
        quad_means = []
        for quad in quadrants:
            if quad.size > 0:
                quad_hsv = cv2.cvtColor(quad, cv2.COLOR_BGR2HSV)
                quad_means.extend(np.mean(quad_hsv, axis=(0, 1)) / 255)
            else:
                quad_means.extend([0, 0, 0])
        
        features['spatial_dist'] = np.array(quad_means)
        
        # 4. Edge features (to detect patterns/logos)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.array([np.mean(edges) / 255])
        
        return features
    
    def compute_similarity(self, features1: Dict[str, np.ndarray], 
                          features2: Dict[str, np.ndarray]) -> float:
        """
        Compute similarity between two feature sets
        """
        similarities = []
        
        # Histogram correlation
        for hist_name in ['h_hist', 's_hist', 'v_hist']:
            if hist_name in features1 and hist_name in features2:
                corr = cv2.compareHist(features1[hist_name].astype(np.float32), 
                                     features2[hist_name].astype(np.float32), 
                                     cv2.HISTCMP_CORREL)
                similarities.append(corr)
        
        # Statistical feature distance
        for feat_name in ['hsv_mean', 'hsv_std', 'spatial_dist']:
            if feat_name in features1 and feat_name in features2:
                dist = np.linalg.norm(features1[feat_name] - features2[feat_name])
                similarity = 1.0 / (1.0 + dist)
                similarities.append(similarity)
        
        # Edge density similarity
        if 'edge_density' in features1 and 'edge_density' in features2:
            edge_diff = abs(features1['edge_density'] - features2['edge_density'])
            similarities.append(1.0 - edge_diff[0])
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_inter_team_similarity(self) -> float:
        """Compute average similarity between the two teams' examples"""
        if 0 not in self.team_examples or 1 not in self.team_examples:
            return 0.0
        
        similarities = []
        for ex0 in self.team_examples[0].examples:
            for ex1 in self.team_examples[1].examples:
                sim = self.compute_similarity(ex0.features, ex1.features)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def initialize_from_user_selection(self, 
                                     frame: np.ndarray,
                                     detections: sv.Detections) -> bool:
        """
        Get user to select multiple players from each team
        Returns True if successful, False if user cancelled
        """
        print("\n=== Interactive Team Selection ===")
        print(f"Click on {self.min_examples_per_team}-{self.max_examples_per_team} players from each team.")
        print("This helps the system learn to distinguish teams accurately.")
        print("Click players to select/deselect them.\n")
        
        # Get white/away team selections
        white_indices = self.get_user_team_selections(frame, detections, "WHITE/AWAY team")
        if not white_indices:
            print("Team selection cancelled.")
            return False
        
        # Store white team examples
        self.team_examples[0] = TeamExamples(team_id=0, examples=[])
        for idx in white_indices:
            bbox = detections.xyxy[idx]
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            
            example = TeamExample(
                team_id=0,
                crop=crop,
                features=self.extract_features(crop),
                bbox=bbox,
                frame=frame
            )
            self.team_examples[0].add_example(example)
        
        print(f"Selected {len(white_indices)} white/away team players")
        
        # Get colored/home team selections
        colored_indices = self.get_user_team_selections(frame, detections, "COLORED/HOME team")
        if not colored_indices:
            print("Team selection cancelled.")
            return False
        
        # Store colored team examples
        self.team_examples[1] = TeamExamples(team_id=1, examples=[])
        for idx in colored_indices:
            bbox = detections.xyxy[idx]
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            
            example = TeamExample(
                team_id=1,
                crop=crop,
                features=self.extract_features(crop),
                bbox=bbox,
                frame=frame
            )
            self.team_examples[1].add_example(example)
        
        print(f"Selected {len(colored_indices)} colored/home team players")
        
        # Verify selections are different enough
        avg_similarity = self._compute_inter_team_similarity()
        
        if avg_similarity > 0.75:
            print(f"WARNING: Teams look similar (avg similarity: {avg_similarity:.2f})")
            print("Consider re-selecting more distinct examples.")
        else:
            print(f"Teams selected successfully! (avg similarity: {avg_similarity:.2f})")
        
        # Show visualization of selections
        vis = self.visualize_examples()
        if vis is not None:
            cv2.imshow("Team Examples", vis)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyWindow("Team Examples")
        
        return True
    
    def predict(self, crops: List[np.ndarray], 
               tracker_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Classify crops based on similarity to user-selected examples
        """
        if len(self.team_examples) < 2:
            raise ValueError("Must initialize with user selection first!")
        
        predictions = []
        
        for i, crop in enumerate(crops):
            # Extract features
            crop_features = self.extract_features(crop)
            
            # Compare to all examples from each team
            team_similarities = {}
            for team_id, team_exs in self.team_examples.items():
                # Compute similarity to each example
                sims = []
                for example in team_exs.examples:
                    sim = self.compute_similarity(crop_features, example.features)
                    sims.append(sim)
                
                # Use max similarity (best match) or average
                # Max is better for handling variety within a team
                team_similarities[team_id] = max(sims) if sims else 0.0
            
            # Predict based on highest similarity
            predicted_team = max(team_similarities, key=team_similarities.get)
            confidence = team_similarities[predicted_team]
            
            # Apply temporal consistency if available
            if tracker_ids is not None and i < len(tracker_ids):
                tracker_id = tracker_ids[i]
                if tracker_id is not None:
                    tracker_id = int(tracker_id)
                    
                    # Add to history
                    self.player_history[tracker_id].append(predicted_team)
                    
                    # Keep recent history
                    if len(self.player_history[tracker_id]) > 10:
                        self.player_history[tracker_id] = self.player_history[tracker_id][-10:]
                    
                    # Use majority vote if enough history and confidence is low
                    if len(self.player_history[tracker_id]) >= 5 and confidence < self.confidence_threshold:
                        team_counts = defaultdict(int)
                        for team in self.player_history[tracker_id]:
                            team_counts[team] += 1
                        predicted_team = max(team_counts, key=team_counts.get)
            
            predictions.append(predicted_team)
        
        return np.array(predictions)
    
    def visualize_examples(self) -> np.ndarray:
        """
        Create a visualization of all selected team examples
        """
        if len(self.team_examples) < 2:
            return None
        
        target_height = 100
        gap = 10
        
        # Process each team's examples
        team_images = []
        for team_id in [0, 1]:
            team_crops = []
            for example in self.team_examples[team_id].examples:
                crop = example.crop
                scale = target_height / crop.shape[0]
                resized = cv2.resize(crop, None, fx=scale, fy=scale)
                team_crops.append(resized)
            
            # Concatenate horizontally
            if team_crops:
                team_row = np.hstack([
                    np.hstack([crop, np.ones((target_height, gap, 3), dtype=np.uint8) * 255])
                    for crop in team_crops[:-1]
                ] + [team_crops[-1]])
                team_images.append(team_row)
        
        # Find max width
        max_width = max(img.shape[1] for img in team_images)
        
        # Pad to same width
        for i, img in enumerate(team_images):
            if img.shape[1] < max_width:
                padding = np.ones((img.shape[0], max_width - img.shape[1], 3), dtype=np.uint8) * 255
                team_images[i] = np.hstack([img, padding])
        
        # Create canvas
        canvas_height = target_height * 2 + gap * 3 + 60
        canvas = np.ones((canvas_height, max_width, 3), dtype=np.uint8) * 255
        
        # Place team examples
        y_offset = 30
        canvas[y_offset:y_offset+target_height, :] = team_images[0]
        canvas[y_offset+target_height+gap:y_offset+target_height*2+gap, :] = team_images[1]
        
        # Add labels
        cv2.putText(canvas, f"Team 0 (White/Away) - {len(self.team_examples[0].examples)} examples", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(canvas, f"Team 1 (Colored/Home) - {len(self.team_examples[1].examples)} examples", 
                   (10, y_offset+target_height+gap-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return canvas