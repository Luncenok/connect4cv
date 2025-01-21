# %% [markdown]
# ## Dice Detection Module
# This module handles the detection of dice values.

# %%
import cv2
import numpy as np
from typing import List, Optional, Tuple
from debug_utils import DebugVisualizer


def detect_edges(frame: np.ndarray, config: dict = None, corners: np.ndarray = None, debug: DebugVisualizer = None) -> np.ndarray:
    """Detect edges in the frame using color masking for white dice."""
    dice_config = config or {}

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if debug:
        debug.show("6.0 Dice - HSV", hsv)
    
    # Create grid mask if corners are provided
    grid_mask = None
    if corners is not None:
        # Calculate center point
        center = np.mean(corners, axis=0)
        
        # Expand corners by 10%
        expanded_corners = np.array([
            center + (corner - center) * 1.4
            for corner in corners
        ])
        
        # Create mask with expanded corners
        grid_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        corners_int = expanded_corners.astype(np.int32)
        cv2.fillPoly(grid_mask, [corners_int], 255)
        # if debug:
        #     debug.show("6.0 Dice - Grid Mask", grid_mask)
    
    # Get color range from config
    lower = np.array(dice_config.get('color_range', {}).get('lower', [0, 0, 240]))
    upper = np.array(dice_config.get('color_range', {}).get('upper', [0, 0, 255]))
    
    # Create white mask
    white_mask = cv2.inRange(hsv, lower, upper)
    # if debug:
    #     debug.show("6.1 Dice - White Mask", white_mask)
    
    # Exclude grid area if mask is available
    if grid_mask is not None:
        white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(grid_mask))
        if debug:
            debug.show("6.2 Dice - White Mask", white_mask)
    
    # Apply morphological operations to clean up the mask
    # kernel = np.ones((5,5), np.uint8)
    # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((10,10), np.uint8)
    # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    if debug:
        debug.show("6.3 Dice - Cleaned Mask", white_mask)
    
    # Edge detection
    canny_low = dice_config.get('canny_low', 50)
    canny_high = dice_config.get('canny_high', 150)
    edges = cv2.Canny(white_mask, canny_low, canny_high)
    if debug:
        debug.show("6.4 Dice - Edges", edges)
    
    return edges


def detect_dice_contours(edges: np.ndarray, frame: np.ndarray = None, debug: DebugVisualizer = None) -> List[np.ndarray]:
    """Detect contours of potential dice in the frame."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    filtered_contours = []
    if debug and frame is not None:
        contour_viz = frame.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Skip small contours
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > 5000:  # Skip large contours
            continue
        aspect_ratio = float(w) / h
        if not (0.8 <= aspect_ratio <= 1.5):  # Skip non-square contours
            continue
            
        filtered_contours.append(contour)
        if debug and frame is not None:
            cv2.drawContours(contour_viz, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(contour_viz, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    if debug and frame is not None:
        debug.show("6.4 Dice - Detected Contours", contour_viz)
    
    if not filtered_contours:
        # return contour of whole frame
        filtered_contours = [np.array([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])]
    return filtered_contours


def extract_dice_region(frame: np.ndarray, contour: np.ndarray, debug: DebugVisualizer = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract dice region from frame using contour."""
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add padding but ensure we stay within image bounds
    padding = 100
    height, width = frame.shape[:2]
    
    y_start = max(0, y - padding)
    y_end = min(height, y + h + padding)
    x_start = max(0, x - padding)
    x_end = min(width, x + w + padding)
    
    dice_region = frame[y_start:y_end, x_start:x_end]
    
    if debug:
        region_viz = frame.copy()
        cv2.rectangle(region_viz, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        debug.show("6.5 Dice - Extracted Region", region_viz)
        debug.show("6.6 Dice - Region Close-up", dice_region)
    
    return dice_region, (x, y, w, h)


def detect_dots(dice_region: np.ndarray, debug: DebugVisualizer = None) -> int:
    """Detect number of dots on the dice."""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(dice_region, cv2.COLOR_BGR2GRAY)
    if debug:
        debug.show("6.7 Dice - Region Grayscale", gray)

    # guassian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if debug:
        debug.show("6.8 Dice - Region Blurred", blur)
    
    _, thresh = cv2.threshold(blur, 205, 255, cv2.THRESH_BINARY_INV)
    if debug:
        debug.show("6.9 Dice - Region Threshold", thresh)

    # detect and remove large areas from thresh. do not use cv2 contours here
    objects = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    for i in range(1, objects[0]):
        if objects[2][i, cv2.CC_STAT_AREA] > 100 or objects[2][i, cv2.CC_STAT_AREA] < 35:
            thresh[objects[1] == i] = 0
    if debug:
        debug.show("6.10 Dice - Region Cleaned", thresh)

    
    # Find contours of dots
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter dot contours by area and circularity
    dot_count = 0
    min_dot_area = 9
    max_dot_area = 80
    
    if debug:
        dots_viz = dice_region.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_dot_area or area > max_dot_area:
            continue
            
        # Check circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.8:  # Not circular enough
            continue
            
        dot_count += 1
        if debug:
            cv2.drawContours(dots_viz, [contour], -1, (0, 255, 0), 2)
    
    if debug:
        debug.show(f"6.11 Dice - Detected Dots: {dot_count}", dots_viz)
    if 1 <= dot_count <= 6:
        return dot_count
    else:
        _, thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
        if debug:
            debug.show("6.12 Dice - Region Threshold", thresh)

        # detect and remove large areas from thresh. do not use cv2 contours here
        objects = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        for i in range(1, objects[0]):
            if objects[2][i, cv2.CC_STAT_AREA] > 80 or objects[2][i, cv2.CC_STAT_AREA] < 10:
                thresh[objects[1] == i] = 0
        if debug:
            debug.show("6.13 Dice - Region Cleaned", thresh)

        
        # Find contours of dots
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter dot contours by area and circularity
        dot_count = 0
        min_dot_area = 10
        max_dot_area = 80
        
        if debug:
            dots_viz = dice_region.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_dot_area or area > max_dot_area:
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.7:  # Not circular enough
                continue
                
            dot_count += 1
            if debug:
                cv2.drawContours(dots_viz, [contour], -1, (0, 255, 0), 2)
        
        if debug:
            debug.show(f"6.14 Dice - Detected Dots: {dot_count}", dots_viz)
        return dot_count if dot_count == 6 else 0


def detect_dice(frame: np.ndarray, config: dict = None, corners: np.ndarray = None, debug: DebugVisualizer = None) -> Optional[int]:
    """Detect dice and its value in the frame."""
    edges = detect_edges(frame, config, corners, debug)
    contours = detect_dice_contours(edges, frame, debug)
    
    if not contours:
        return None

    # sort largest to smallest
    contours.sort(key=cv2.contourArea, reverse=False)
    
    # Process each potential dice
    for contour in contours:
        dice_region, _ = extract_dice_region(frame, contour, debug)
        dots = detect_dots(dice_region, debug)
        if dots > 0:
            return dots
    
    return None


def visualize_dice_detection(frame: np.ndarray, dice_value: Optional[int], debug: DebugVisualizer = None) -> np.ndarray:
    """Visualize dice detection results."""
    if dice_value is None:
        return frame
        
    # Draw dice value
    viz_frame = frame.copy()
    height, width = viz_frame.shape[:2]
    cv2.putText(viz_frame, f"Dice: {dice_value}",
                (width*3//4, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    if debug:
        debug.show("6.11 Dice - Final Result", viz_frame)
    
    return viz_frame