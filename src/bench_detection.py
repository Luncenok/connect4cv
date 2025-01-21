import cv2
import numpy as np
from typing import Tuple, Optional, List
from debug_utils import DebugVisualizer

def detect_bench_counters(frame: np.ndarray, 
                         corners: np.ndarray,
                         yellow_mask: np.ndarray,
                         red_mask: np.ndarray,
                         debug: Optional[DebugVisualizer] = None) -> Tuple[Tuple[int, List[Tuple[int, int]]], Tuple[int, List[Tuple[int, int]]]]:
    """
    Detect and count red and yellow counters outside the grid (on the bench).
    
    Args:
        frame: Input frame
        corners: Grid corner points
        yellow_mask: Binary mask for yellow counters
        red_mask: Binary mask for red counters
        debug: Optional debug visualizer
    
    Returns:
        Tuple of ((yellow_count, yellow_centers), (red_count, red_centers))
    """
    # Create and expand grid mask
    height, width = frame.shape[:2]
    grid_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Expand corners slightly to ensure we cover the entire grid
    center = np.mean(corners, axis=0)
    expanded_corners = corners + (corners - center) * 0.1
    corners_int = expanded_corners.astype(np.int32)
    
    # Create grid mask
    cv2.fillPoly(grid_mask, [corners_int], 255)
    # if debug:
    #     debug.show("7.0 Bench - Grid Mask", grid_mask)
    
    # Remove grid area from counter masks
    yellow_bench = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(grid_mask))
    red_bench = cv2.bitwise_and(red_mask, cv2.bitwise_not(grid_mask))
    
    if debug:
        debug.show("7.1 Bench - Yellow Mask", yellow_bench)
        debug.show("7.2 Bench - Red Mask", red_bench)
    
    # Process each color mask
    def count_counters(mask: np.ndarray, title: str = "", min_area: int = 400, max_area: int = 20000) -> Tuple[int, List[Tuple[int, int]]]:
        # Clean up the mask
        if title == "red":
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        else:
            kernel = np.ones((10,10), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # kernel = np.ones((15,15), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if debug and contours:
            viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)
            debug.show(f"7.3 Bench {title} - All Detected Counters ({len(contours)})", viz)
        
        # Filter contours by area and circularity
        counter_count = 0
        valid_contours = []
        centers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # Not circular enough
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
                
            counter_count += 1
            valid_contours.append(contour)
        
        if debug and valid_contours:
            viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(viz, valid_contours, -1, (0, 255, 0), 2)
            for i, (cx, cy) in enumerate(centers, 1):
                cv2.putText(viz, str(i), (cx-10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            debug.show(f"7.4 Bench {title} - Detected Counters ({counter_count})", viz)
        
        return counter_count, centers
    
    yellow_result = count_counters(yellow_bench, "yellow")
    red_result = count_counters(red_bench, "red")
    
    return yellow_result, red_result
