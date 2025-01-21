# Connect Four Game Detection System

Mateusz Idziejczak 155842<br>
Mateusz Stawicki 155900

## Game Rules
Connect Four is a two-player connection board game where players take turns dropping colored counters into a vertically suspended grid. The objective is to be the first to form a horizontal, vertical, or diagonal line of four counters of the same color. Our variantt is played horizontally on ground

Specific rules for our implementation:
1. The game board is 6x7 (6 rows, 7 columns)
2. Players use red and yellow counters
3. First roll determines who goes first. highest roll goes first. If tie roll again.
4. First dice roll of a player determines how many counters a player can place in their turn (1,2 - 0; 3,4 - 1; 5,6 - 2)
5. A player can place a counter in any empty column
6. A second dice roll determines how many counters a player can take to their bench (1,2 - 0; 3,4 - 1; 5,6 - 2)
7. If player has no counters left in bench he skips first roll and only rolls to take more counters
8. The game starts when the first valid dice value is detected
9. A player wins by connecting four counters of their color in any direction

## Dataset Description

The dataset was specifically designed to challenge computer vision algorithms with several deliberate complexities:

### Challenging Elements

1. **Board Design**:
   - Hand-colored grid cells creating non-uniform background
   - Irregular grid lines and cell shapes

2. **Dice Area**:
   - Zebra-striped white pattern under the white dice creating noise
   - Multiple white objects in the scene that could be confused with dice

3. **Counter Characteristics**:
   - Counters with text/logos adding noise
   - Irregular counter shapes

## Detected events

1. Game start (first roll)
2. Player move
3. Dice roll (dot count)
4. Bench counter update
5. Game end (winning condition detected)

## Technical Implementation

### 1. Preprocessing

The preprocessing pipeline consists of several key steps to prepare each frame for analysis:

1. **Frame Stabilization**:
   - Convert frame to grayscale
   - Track feature points between frames using goodFeaturesToTrack
   - Calculate optical flow using calcOpticalFlowPyrLK
   - Estimate affine transformation matrix

2. **Noise Reduction**:
   - Apply 5x5 Gaussian blur
   - Helps reduce high-frequency noise

3. **Lighting Normalization**:
   - Convert to LAB color space
   - Apply CLAHE to L channel (clipLimit=2.0, tileGridSize=8x8)
   - Convert back to BGR
   - Results in more uniform lighting across the frame

### 2. Grid Detection

The grid detection process involves several steps to accurately locate and analyze the game board:

1. **Grid Mask Creation**:
   - Convert frame to HSV color space
   - Apply color thresholding using grid_lower and grid_upper bounds
   - Create binary mask for grid area

2. **Grid Contour Detection**:
   - Find all external contours
   - Filter contours by area (> 50000 pixels)

3. **Corner Detection and Sorting**:
   - Use adaptive epsilon factors (0.01-0.08) for corner approximation
   - Find exactly 4 corners using approxPolyDP
   - Sort corners by angle from center
   - Rotate array to ensure top-left corner is first

4. **Cell Center Calculation**:
   - Create perspective transform matrix
   - Calculate centers in normalized space
   - Transform back to original image coordinates

5. **Counter Color Detection**:
   - Extract ROI around center point
   - Create red and yellow color masks
   - Calculate color coverage ratios
   - Classify as RED/YELLOW/EMPTY based on threshold (0.3)

The system uses HSV color space for robust color detection and handles perspective distortion through transformation matrices. Cell analysis uses relative sizing (40% of cell size) to adapt to different board sizes and camera angles.

### 4. Dice Detection

The dice detection system uses a multi-stage approach to identify and read dice values:

1. **Edge Detection and Masking**:
   - Convert frame to HSV
   - Create white color mask ([0,0,240] to [0,0,255])
   - Exclude grid area using expanded grid mask
   - Apply Canny edge detection

2. **Dice Region Extraction**:
   - Detect contours in edge image
   - Filter contours by:
     * Area (100-5000 pixels)
     * Aspect ratio (0.8-1.5)
   - Add padding around detected regions
   - When no region found return whole frame

3. **Dot Detection**:
   - Convert to grayscale and apply Gaussian blur
   - Apply binary threshold (205)
   - Remove large connected components (>100 pixels)
   - Filter dot contours by:
     * Area (35-100 pixels)
     * Circularity (>0.8)
   - If no dots found, retry with lower threshold (110) to detect 6 

4. **Value Validation**:
   - Verify detected dots
   - Count dots that meet all criteria
   - Validate count is between 1-6
   - Return detected value or None

The system handles the zebra-striped background by using strict white color thresholding and robust dot detection parameters. Multiple threshold attempts help handle varying lighting conditions.

### 5. Bench Counter Detection

The bench counter detection system identifies and tracks counters outside the main grid:

1. **Grid Exclusion**:
   - Create grid mask
   - Calculate grid center point
   - Expand grid corners by 10%
   - Create binary mask of expanded grid area
   - Remove grid area from counter masks

2. **Counter Detection**:
   - Process each color separately
   - Apply morphological operations
   - Find external contours

3. **Counter Validation**:
   - Filter detected counters
   - Area constraints:
     * Minimum: 400 pixels
     * Maximum: 20000 pixels
   - Circularity check (>0.5)
   - Calculate center using moments

The system handles different counter colors with adapted morphological operations, accounting for their unique characteristics in the image. The expanded grid mask ensures no grid counters are mistakenly counted as bench counters.

### 5. Game State Management

The game state management system consists of several components working together:

1. **Grid State Tracking**:
   - 6x7 grid array for counter positions
   - Permanent counter tracking

2. **Win Detection**:
   - Check all winning patterns
   - Horizontal (4 in a row)
   - Vertical (4 in a column)
   - Diagonal positive slope
   - Diagonal negative slope
   - Only consider permanent counters

3. **Event Logging**:
   - Game start/end
   - Player moves
   - Bench counter updates
   - Dice value changes
   - Save to JSON with timestamps

4. **Dice Processing**:
   - Temporal stability for dice values
   - Maintain 30-frame history
   - Require 10 consecutive frames
   - Clear value if stability lost
   - Start game on first stable value

5. **Move Validation**:
   - if counter in more than 50 frames then it is marked permament
   - Update grid state
   - Check for winning patterns
   - Detect draw conditions
   - Log game events

6. **Bench Processing**:
   - Maintain 120-frame history
   - Cluster nearby positions (20px tolerance)
   - Require 30 consecutive detections
   - Track yellow and red counters separately

The system ensures game stability by requiring consistent detection before confirming moves or dice values. Win conditions are only checked using permanent counters, and all game events are logged with timestamps for analysis.

### System Effectiveness

### Grid Detection
- Main challenges:
  - blue dice tray interferences with grid detection
  - Severe camera angles
  - Detecting grid at the end is hard as counters cover grid

### Counter Detection
- Main challenges:
  - Overlapping counters on bench
  - Counter variations

### Dice Detection
- Main challenges:
  - Zebra-striped white background 
  - when 6 is rolled dots are close together

## Analysis and Conclusions

### Strengths
1. Robust grid detection even with irregular coloring
2. Reliable counter color classification
3. Stable game state tracking
4. Noise handling in dice detection

### Limitations
1. Sensitivity to extreme lighting conditions
2. Occasional false positives in dice detection
3. Counter detection issues with significant overlap
4. Grid is detected from first 10 frames. When grid is moved position is lost

### Future Improvements
1. Better parameters for dice detection
2. Dynamic parameter adjustment based on conditions
