# %% [markdown]
# ## Preprocessing Module
# This module handles video preprocessing steps.

# %%
import cv2
import numpy as np
from typing import Optional, Tuple

from debug_utils import DebugVisualizer

def estimate_transform(frame: np.ndarray, 
                      prev_gray: Optional[np.ndarray] = None,
                      prev_points: Optional[np.ndarray] = None,
                      debug_visualizer: Optional[DebugVisualizer] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate transformation matrix for frame stabilization."""
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if debug_visualizer:
        debug_visualizer.show("1.1 Grayscale for Stabilization", gray)
    
    # Initialize or update feature points
    if prev_gray is None or prev_points is None:
        points = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        if debug_visualizer:
            points_viz = frame.copy()
            points_viz = debug_visualizer.draw_points(points_viz, points[:, 0])
            debug_visualizer.show("1.2 Initial Feature Points", points_viz)
        return np.eye(2, 3, dtype=np.float32), gray, points
    
    if debug_visualizer:
        points_viz = frame.copy()
        points_viz = debug_visualizer.draw_points(points_viz, prev_points[:, 0])
        debug_visualizer.show("1.2 Initial Feature Points", points_viz)
    
    # Calculate optical flow
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_points, None)
    
    # Filter good points
    good_old = prev_points[status == 1]
    good_new = curr_points[status == 1]
    
    if len(good_old) < 2 or len(good_new) < 2:
        points = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        return np.eye(2, 3, dtype=np.float32), gray, points
    
    # Show tracked points
    if debug_visualizer:
        flow_viz = frame.copy()
        for old, new in zip(good_old, good_new):
            x1, y1 = old.ravel()
            x2, y2 = new.ravel()
            cv2.line(flow_viz, (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 255, 0), 2)
            cv2.circle(flow_viz, (int(x2), int(y2)), 3, (0, 0, 255), -1)
        debug_visualizer.show("1.3 Optical Flow", flow_viz)
    
    # Estimate transformation
    transform = cv2.estimateAffinePartial2D(good_old, good_new)[0]
    
    # Get new points for next frame
    points = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    
    return transform if transform is not None else np.eye(2, 3, dtype=np.float32), gray, points

def apply_gaussian(frame: np.ndarray, debug_visualizer: Optional[DebugVisualizer] = None) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred

def apply_clahe(frame: np.ndarray, debug_visualizer: Optional[DebugVisualizer] = None) -> np.ndarray:
    """Apply CLAHE for lighting adjustment."""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    if debug_visualizer:
        debug_visualizer.show("1.6 LAB Color Space", lab)
    
    # Split channels
    l, a, b = cv2.split(lab)
    if debug_visualizer:
        debug_visualizer.show("1.7 L Channel", l)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    if debug_visualizer:
        debug_visualizer.show("1.8 CLAHE L Channel", cl)
    
    # Merge channels
    adjusted_lab = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    adjusted = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    
    return adjusted
