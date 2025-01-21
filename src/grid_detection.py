# %% [markdown]
# ## Grid Detection Module
# This module handles the detection of the game grid and the current state of counters.

# %%
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from enum import Enum
from debug_utils import DebugVisualizer

class CounterState(Enum):
    EMPTY = 0
    RED = 1
    YELLOW = 2

def create_grid_mask(frame: np.ndarray, 
                    grid_lower: List[int], 
                    grid_upper: List[int]) -> np.ndarray:
    """Create a mask for the grid using color thresholding."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grid_lower = np.array(grid_lower)
    grid_upper = np.array(grid_upper)
    grid_mask = cv2.inRange(hsv, grid_lower, grid_upper)
    
    return grid_mask

def find_largest_contour(grid_mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the grid contour by selecting the second largest when there are two large contours."""
    # Find all contours
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Filter contours larger than 50000 and sort by area
    large_contours = [c for c in contours if cv2.contourArea(c) > 50000]
    large_contours.sort(key=cv2.contourArea, reverse=True)
    
    # If we have exactly 2 large contours, return the second one (the grid)
    if len(large_contours) == 2:
        return large_contours[1]
    
    # Otherwise, just return the largest contour
    return max(contours, key=cv2.contourArea)

def find_corner_points(contour: np.ndarray, epsilon_factor: float = 0.02) -> Optional[np.ndarray]:
    # Try a range of epsilon factors to find exactly 4 corners
    epsilon_factors = [epsilon_factor * i for i in [0.5, 1.0, 2.0, 4.0]]
    
    for eps_factor in epsilon_factors:
        epsilon = eps_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            return np.float32(approx[:, 0, :])
    
    return None



def sort_corners(corners: np.ndarray) -> np.ndarray:
    # Calculate the center point
    center = np.mean(corners, axis=0)
    
    # Calculate vectors from center to corners
    vectors = corners - center
    
    # Calculate angles between vectors and positive x-axis
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # Sort corners by angle
    sorted_indices = np.argsort(angles)
    corners = corners[sorted_indices]
    
    # Rotate array so that top-left corner (most negative x and y) is first
    distances = np.sum((corners - np.min(corners, axis=0)) ** 2, axis=1)
    rotation = np.argmin(distances)
    if rotation > 0:
        corners = np.roll(corners, -rotation, axis=0)
    
    return corners

def get_cell_centers(grid_corners: np.ndarray, rows: int, cols: int) -> List[Tuple[int, int]]:
    # Create source points (actual grid corners)
    src_pts = grid_corners.astype(np.float32)
    
    # Create destination points (normalized rectangular grid)
    width, height = 100 * cols, 100 * rows  # arbitrary scale
    dst_pts = np.array([
        [0, 0],              # Top-left
        [width, 0],          # Top-right
        [width, height],     # Bottom-right
        [0, height]          # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform
    perspective_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    # Generate normalized grid cell centers
    centers = []
    cell_width = width / cols
    cell_height = height / rows
    
    # For each cell in the normalized grid
    for row in range(rows):
        for col in range(cols):
            # Calculate center in normalized space
            x = int((col + 0.5) * cell_width)
            y = int((row + 0.5) * cell_height)
            
            # Transform point back to original image space
            pt = np.array([[[x, y]]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(pt, perspective_matrix)
            
            # Add to centers list as integer coordinates
            center_x = int(transformed_pt[0][0][0])
            center_y = int(transformed_pt[0][0][1])
            centers.append((center_x, center_y))
    
    return centers

def create_color_masks(roi: np.ndarray,
                      red_lower1: List[int],
                      red_upper1: List[int],
                      red_lower2: List[int],
                      red_upper2: List[int],
                      yellow_lower: List[int],
                      yellow_upper: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create red and yellow color masks for counter detection."""
    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Create color masks
    red_mask1 = cv2.inRange(hsv, np.array(red_lower1), np.array(red_upper1))
    red_mask2 = cv2.inRange(hsv, np.array(red_lower2), np.array(red_upper2))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, np.array(yellow_lower), np.array(yellow_upper))
    
    return red_mask, yellow_mask

def detect_counter_color(frame: np.ndarray,
                        center: Tuple[int, int],
                        radius: int,
                        red_lower1: List[int],
                        red_upper1: List[int],
                        red_lower2: List[int],
                        red_upper2: List[int],
                        yellow_lower: List[int],
                        yellow_upper: List[int],
                        cols: int,
                        rows: int,
                        color_threshold: float = 0.3,
                        corners: Optional[np.ndarray] = None,
                        pre_calculated_masks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        debug_visualizer: Optional[DebugVisualizer] = None) -> CounterState:

    left, top = corners[0]
    right, bottom = corners[2]
    cell_width = (right - left) / cols
    cell_height = (bottom - top) / rows
    radius = int(min(cell_width, cell_height) * 0.4)  # Same as visualization
    
    # Extract ROI
    x, y = center
    roi = frame[y-radius:y+radius, x-radius:x+radius]
    
    # Get color masks for ROI
    red_mask, yellow_mask = pre_calculated_masks
    # Extract the masks for this ROI
    red_mask_roi = red_mask[y-radius:y+radius, x-radius:x+radius]
    yellow_mask_roi = yellow_mask[y-radius:y+radius, x-radius:x+radius]

    # Calculate coverage ratios
    total_pixels = np.prod(roi.shape[:2])
    red_ratio = np.count_nonzero(red_mask_roi) / total_pixels
    yellow_ratio = np.count_nonzero(yellow_mask_roi) / total_pixels
    
    if debug_visualizer:
        roi_viz = frame.copy()
        # Draw square ROI
        cv2.rectangle(roi_viz, 
                     (x-radius, y-radius), 
                     (x+radius, y+radius), 
                     (0, 255, 0), 2)
        debug_visualizer.show(f"Counter ROI {x}, {y}", roi_viz)
        debug_visualizer.show(f"Counter ROI Zoomed {x}, {y} - Red {red_ratio} - Yellow {yellow_ratio}", roi)
    
    # Determine counter color
    if red_ratio > color_threshold:
        return CounterState.RED
    elif yellow_ratio > color_threshold:
        return CounterState.YELLOW
    else:
        return CounterState.EMPTY
