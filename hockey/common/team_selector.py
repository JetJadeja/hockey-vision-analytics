import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TeamSelection:
    """Stores the result of team selection."""
    team_names: Dict[int, str]  # {0: "TOR", 1: "DET"}
    selected_players: Dict[int, List[int]]  # {0: [tracker_ids], 1: [tracker_ids]}


class InteractiveTeamSelector:
    """
    Interactive UI for selecting players from each team and naming teams.
    """
    
    def __init__(self):
        self.selected_players = {0: [], 1: []}
        self.team_names = {0: "", 1: ""}
        self.current_team = 0
        self.frame = None
        self.detections = None
        self.display_frame = None
        self.selection_complete = False
        self.clicked_points = []
        
    def select_teams(
        self, 
        frame: np.ndarray, 
        detections: sv.Detections,
        window_name: str = "Team Selection"
    ) -> Optional[TeamSelection]:
        """
        Open interactive window for team selection.
        
        Args:
            frame: Frame with players visible
            detections: Player detections (must have tracker_id)
            window_name: Name of the selection window
            
        Returns:
            TeamSelection object with team names and assignments
        """
        if detections.tracker_id is None:
            print("Error: Detections must have tracker IDs")
            return None
            
        self.frame = frame.copy()
        self.detections = detections
        self.display_frame = frame.copy()
        
        # Set up mouse callback
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # Start with team 0 selection
        self._update_display()
        
        print("\n=== TEAM SELECTION ===")
        print("Click on 2-3 players from the HOME team")
        print("Press SPACE when done selecting")
        print("Press ESC to cancel")
        
        while not self.selection_complete:
            cv2.imshow(window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return None
            elif key == 32:  # SPACE
                if self.current_team == 0 and len(self.selected_players[0]) > 0:
                    # Get home team name
                    team_name = self._get_team_name("HOME")
                    if team_name:
                        self.team_names[0] = team_name
                        self.current_team = 1
                        self.clicked_points = []
                        print(f"\nHome team set as: {team_name}")
                        print("Now click on 2-3 players from the AWAY team")
                        self._update_display()
                elif self.current_team == 1 and len(self.selected_players[1]) > 0:
                    # Get away team name
                    team_name = self._get_team_name("AWAY")
                    if team_name:
                        self.team_names[1] = team_name
                        self.selection_complete = True
                        print(f"Away team set as: {team_name}")
        
        cv2.destroyWindow(window_name)
        
        if self.selection_complete:
            return TeamSelection(
                team_names=self.team_names,
                selected_players=self.selected_players
            )
        return None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for player selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find which detection was clicked
            clicked_idx = self._find_clicked_detection(x, y)
            if clicked_idx is not None:
                tracker_id = self.detections.tracker_id[clicked_idx]
                
                # Toggle selection
                if tracker_id in self.selected_players[self.current_team]:
                    self.selected_players[self.current_team].remove(tracker_id)
                else:
                    self.selected_players[self.current_team].append(tracker_id)
                
                self._update_display()
    
    def _find_clicked_detection(self, x: int, y: int) -> Optional[int]:
        """Find which detection contains the clicked point."""
        for i, xyxy in enumerate(self.detections.xyxy):
            x1, y1, x2, y2 = xyxy
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None
    
    def _update_display(self):
        """Update the display frame with current selections."""
        self.display_frame = self.frame.copy()
        
        # Draw all detections
        for i, (xyxy, tracker_id) in enumerate(zip(self.detections.xyxy, self.detections.tracker_id)):
            x1, y1, x2, y2 = xyxy.astype(int)
            
            # Determine color based on selection
            if tracker_id in self.selected_players[0]:
                color = (0, 255, 0)  # Green for home team
                thickness = 3
            elif tracker_id in self.selected_players[1]:
                color = (0, 0, 255)  # Red for away team
                thickness = 3
            else:
                color = (128, 128, 128)  # Gray for unselected
                thickness = 1
            
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw tracker ID
            cv2.putText(
                self.display_frame,
                str(tracker_id),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Draw instructions
        instructions = []
        if self.current_team == 0:
            instructions.append("Select HOME team players (Green)")
            instructions.append(f"Selected: {len(self.selected_players[0])} players")
        else:
            instructions.append("Select AWAY team players (Red)")
            instructions.append(f"Selected: {len(self.selected_players[1])} players")
        instructions.append("Press SPACE when done, ESC to cancel")
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                self.display_frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # Draw background for better readability
            (text_width, text_height), _ = cv2.getTextSize(
                instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                self.display_frame,
                (8, y_offset - text_height - 2),
                (12 + text_width, y_offset + 2),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                self.display_frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y_offset += 30
    
    def _get_team_name(self, team_type: str) -> Optional[str]:
        """Get team name via simple text input."""
        # Create a simple input window
        window_name = f"Enter {team_type} Team Name"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 100)
        
        # Create black image for text input
        input_img = np.zeros((100, 400, 3), dtype=np.uint8)
        team_name = ""
        
        print(f"\nEnter {team_type} team name/abbreviation (e.g., TOR, MTL, DET):")
        print("Press ENTER when done")
        
        while True:
            display_img = input_img.copy()
            cv2.putText(
                display_img,
                f"{team_type} Team: {team_name}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.imshow(window_name, display_img)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == 13:  # ENTER
                cv2.destroyWindow(window_name)
                return team_name if team_name else None
            elif key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return None
            elif key == 8:  # BACKSPACE
                team_name = team_name[:-1]
            elif 32 <= key <= 126:  # Printable characters
                team_name += chr(key)  # Keep original case
                if len(team_name) > 10:  # Limit length
                    team_name = team_name[:10]