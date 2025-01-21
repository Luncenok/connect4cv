# %% [markdown]
# ## Step 1: Data Handling Module
# This module handles video input and frame extraction for the Connect4 game analysis.

# %% [markdown]
# ### Video Input and Frame Extraction

# %%
import cv2
import numpy as np
import os
from pathlib import Path

class VideoHandler:
    def __init__(self, video_path):
        """Initialize video handler with path to video file."""
        self.video_path = Path(video_path)
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        
    def load_video(self):
        """Load video file and get basic metadata."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return self.cap.isOpened()
    
    def extract_frame(self, frame_number):
        """Extract a specific frame from the video."""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Video file not loaded. Call load_video() first.")
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"Could not extract frame {frame_number}")
            
        return frame
    
    def __del__(self):
        """Clean up video capture."""
        if self.cap is not None:
            self.cap.release()

# %% [markdown]
# ### Usage Example

# %%
if __name__ == "__main__":
    # Example usage
    video_handler = VideoHandler("path_to_video.mp4")
    if video_handler.load_video():
        print(f"Loaded video with {video_handler.frame_count} frames at {video_handler.fps} FPS")
        
        # Extract first frame
        frame = video_handler.extract_frame(0)
