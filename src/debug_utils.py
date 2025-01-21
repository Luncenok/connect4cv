# %% [markdown]
# ## Debug Utilities Module
# This module provides utilities for debug visualization.

# %%
import cv2
import numpy as np
from typing import Optional

class DebugVisualizer:
    def __init__(self, enabled: bool = False, streamlit_mode: bool = False):
        """Initialize debug visualizer."""
        self.enabled = enabled
        self.streamlit_mode = streamlit_mode
        self.debug_images = {}
    
    def show(self, title: str, image: np.ndarray) -> None:
        """Store and Show debug image if debug is enabled."""
        if self.enabled and image is not None:
            self.debug_images[title] = image.copy()
            if not self.streamlit_mode:
                cv2.imshow(title, image)
                cv2.waitKey(0)
    
    def get_debug_images(self):
        """Get the dictionary of debug images."""
        return self.debug_images
    
    def clear_debug_images(self):
        """Clear the stored debug images."""
        self.debug_images = {}
    
    def draw_circles(self, image: np.ndarray,
                    circles: Optional[np.ndarray],
                    color: tuple = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
        """Draw circles on image copy."""
        result = image.copy()
        if circles is not None:
            for circle in circles:
                x, y, r = circle
                cv2.circle(result, (x, y), r, color, thickness)
        return result
    
    def draw_lines(self, image: np.ndarray,
                  lines: Optional[np.ndarray],
                  color: tuple = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """Draw lines on image copy."""
        result = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        return result
    
    def draw_points(self, image: np.ndarray,
                   points: np.ndarray,
                   color: tuple = (0, 0, 255),
                   size: int = 5) -> np.ndarray:
        """Draw points on image copy."""
        result = image.copy()
        for point in points:
            cv2.circle(result, tuple(point.astype(int)), size, color, -1)
        return result
