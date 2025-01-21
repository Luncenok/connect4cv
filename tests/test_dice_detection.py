# %% [markdown]
# ## Unit Tests for Dice Detection Module

import pytest
import numpy as np
import cv2
from src.dice_detection import DiceDetector, DiceValueExtractor

# %% [markdown]
# ### Test Setup

@pytest.fixture
def dice_detector():
    """Create a dice detector instance for testing."""
    return DiceDetector()

@pytest.fixture
def value_extractor():
    """Create a value extractor instance for testing."""
    return DiceValueExtractor()

@pytest.fixture
def sample_dice_image():
    """Create a synthetic dice image for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw circles representing pips
    centers = [(25, 25), (75, 75)]  # Two pips for testing
    for center in centers:
        cv2.circle(img, center, 10, (255, 255, 255), -1)
    return img

# %% [markdown]
# ### Test Cases

def test_edge_detection(dice_detector, sample_dice_image):
    """Test edge detection functionality."""
    edges = dice_detector.detect_edges(sample_dice_image)
    assert edges is not None
    assert edges.shape == (100, 100)
    assert np.sum(edges > 0) > 0  # Should detect some edges

def test_circle_detection(dice_detector, sample_dice_image):
    """Test circle detection functionality."""
    edges = dice_detector.detect_edges(sample_dice_image)
    circles = dice_detector.detect_circles(edges, sample_dice_image)
    assert len(circles) == 2  # Should detect two pips

def test_value_extraction(value_extractor, sample_dice_image):
    """Test dice value extraction."""
    value = value_extractor.get_dice_value(sample_dice_image)
    assert value == 2  # Should detect two pips

def test_empty_image(value_extractor):
    """Test handling of empty image."""
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    value = value_extractor.get_dice_value(empty_image)
    assert value == 0  # Should return 0 for empty image
