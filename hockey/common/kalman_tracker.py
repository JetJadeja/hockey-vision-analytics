"""
Kalman filter implementation for smooth bounding box tracking in hockey videos.
"""

import numpy as np
from typing import Tuple, Optional


class KalmanBoxFilter:
    """
    A Kalman filter for tracking bounding boxes in 2D space.
    
    The state space is:
    [x, y, w, h, dx, dy, dw, dh]
    where (x,y) is center position, (w,h) is width/height, and d* are velocities.
    """
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float], dt: float = 1.0):
        """
        Initialize Kalman filter.
        
        Args:
            initial_bbox: (x1, y1, x2, y2) format
            dt: Time step between frames
        """
        # Convert bbox to center format
        x1, y1, x2, y2 = initial_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        # Initialize state vector [x, y, w, h, dx, dy, dw, dh]
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float32)
        
        # State transition matrix
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = dt  # x += dx * dt
        self.F[1, 5] = dt  # y += dy * dt
        self.F[2, 6] = dt  # w += dw * dt
        self.F[3, 7] = dt  # h += dh * dt
        
        # Measurement matrix (we measure x, y, w, h)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.  # x
        self.H[1, 1] = 1.  # y
        self.H[2, 2] = 1.  # w
        self.H[3, 3] = 1.  # h
        
        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32)
        # Position and size have lower noise
        self.Q[0:4, 0:4] *= 0.01
        # Velocity has higher noise (can change more)
        self.Q[4:8, 4:8] *= 0.1
        
        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance matrix
        self.P = np.eye(8, dtype=np.float32) * 10.
        
        # Innovation covariance for adaptive filtering
        self.S = np.eye(4, dtype=np.float32)
        
        # Store last measurement for motion analysis
        self.last_measurement = np.array([cx, cy, w, h], dtype=np.float32)
        
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        
        Returns:
            Predicted state vector
        """
        # Predict state
        self.x = self.F @ self.x
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement: Tuple[float, float, float, float], 
               confidence: float = 1.0) -> np.ndarray:
        """
        Update state with new measurement.
        
        Args:
            measurement: (x1, y1, x2, y2) bbox format
            confidence: Detection confidence [0, 1]
            
        Returns:
            Updated state vector
        """
        # Convert measurement to center format
        x1, y1, x2, y2 = measurement
        z = np.array([
            (x1 + x2) / 2.0,  # cx
            (y1 + y2) / 2.0,  # cy
            x2 - x1,          # w
            y2 - y1           # h
        ], dtype=np.float32)
        
        # Adaptive measurement noise based on confidence
        R_adaptive = self.R / max(confidence, 0.1)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        self.S = self.H @ self.P @ self.H.T + R_adaptive
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update error covariance
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P
        
        # Store measurement for motion analysis
        self.last_measurement = z
        
        return self.x.copy()
    
    def get_state_bbox(self) -> Tuple[float, float, float, float]:
        """
        Get current state as bbox.
        
        Returns:
            Bounding box in (x1, y1, x2, y2) format
        """
        cx, cy, w, h = self.x[:4]
        
        # Ensure positive dimensions
        w = max(w, 1.0)
        h = max(h, 1.0)
        
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        return (x1, y1, x2, y2)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity (dx, dy)."""
        return (self.x[4], self.x[5])
    
    def get_motion_magnitude(self) -> float:
        """Get magnitude of motion vector."""
        dx, dy = self.get_velocity()
        return np.sqrt(dx**2 + dy**2)