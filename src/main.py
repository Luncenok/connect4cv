# %% [markdown]
# ## Main Script
# This script coordinates all the modules to process the game video and detect game events.

# %%
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List

import yaml
from data_handling import VideoHandler
from preprocessing import estimate_transform, apply_gaussian, apply_clahe
from dice_detection import detect_dice
from bench_detection import detect_bench_counters
from debug_utils import DebugVisualizer
from grid_detection import (CounterState, get_cell_centers, create_grid_mask, detect_counter_color,
                      find_largest_contour, find_corner_points, sort_corners, create_color_masks)
from game_state import GameState, GameController, CellState

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Connect4 with Dice game video.')
    parser.add_argument('--video', type=str, default='data/easy1.mp4',
                      help='Path to input video file')
    parser.add_argument('--output', type=str, default='output/game_log.json',
                      help='Path to output log file')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--display', action='store_true',
                      help='Display processed frames')
    parser.add_argument('--debug', type=int,
                      help='Frame number to debug (shows all processing steps)')
    parser.add_argument('--counters', action='store_true',
                      help='Enable counter detection debug visualization')
    args = parser.parse_args()
    
    # Set up paths
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file {video_path} does not exist")
        return
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Global game state
    game_state = GameState()
    game_controller = GameController(game_state, str(output_path))
    
    # Process video
    process_video(
        str(video_path),
        str(output_path),
        args.config,
        debug=args.debug is not None,
        debug_counters=args.counters,
        debug_frame=args.debug,
        display=args.display,
        game_state=game_state,
        game_controller=game_controller
    )

def load_config(config_path: str) -> Dict:
    """Load and parse configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize_video_writer(video_path: str, output_path: str, fps: int) -> Optional[cv2.VideoWriter]:
    """Initialize video writer for output video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    output_video_path = str(Path(output_path).parent / 'annotated_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def process_video(video_path: str,
                 output_path: str,
                 config_path: str = "config.yaml",
                 debug: bool = False,
                 debug_counters: bool = False,
                 debug_frame: Optional[int] = None,
                 display: bool = False,
                 game_state: GameState = None,
                 game_controller: GameController = None) -> None:
    """Process the entire game video."""
    # Load configuration
    config = load_config(config_path)
    video_config = config['video']
    output_config = config['output']
    detection_config = config['detection']
    grid_params = detection_config['grid']
    
    # Constants
    FRAMES_TO_PROCESS = 10  # Number of frames to process for corner averaging
    
    # Initialize components
    video_handler = VideoHandler(video_path)
    if not video_handler.load_video():
        raise RuntimeError("Failed to load video")
    
    debug_visualizer = DebugVisualizer(debug) if debug else None
    
    # Initialize video writer if needed
    video_writer = None
    if output_config.get('save_annotated_video', True):
        fps = video_config.get('frame_rate', 30)
        video_writer = initialize_video_writer(video_path, output_path, fps)
    
    try:
        if debug_frame is not None:
            # Process only the debug frame
            frame = video_handler.extract_frame(debug_frame)
            if frame is not None:
                prev_frame = video_handler.extract_frame(debug_frame - 1) if debug_frame > 0 else None
                # If debug frame is after first frames, calculate average from first frames
                if debug_frame >= FRAMES_TO_PROCESS:
                    corners_history = []
                    prev_gray = None
                    prev_points = None
                    for tqdm_i in tqdm(range(FRAMES_TO_PROCESS), desc=f"Processing first {FRAMES_TO_PROCESS} frames"):
                        i = tqdm_i
                        temp_frame = video_handler.extract_frame(i)
                        if temp_frame is not None:
                            adjusted, transform, gray, points = preprocess_frame(temp_frame, prev_gray, prev_points, None)
                            _, _, corners = process_frame(
                                adjusted, grid_params,
                                debug_visualizer=None,  # No debug visualization for temp frames
                                transform=transform,
                                game_state=game_state,
                                game_controller=game_controller,
                                frame_number=i,
                                detection_config=detection_config
                            )
                            if corners is not None:
                                corners_history.append(corners)
                            prev_frame = temp_frame
                            prev_gray = gray
                            prev_points = points
                    
                    adjusted, _, _, _ = preprocess_frame(frame, prev_gray, prev_points, debug_visualizer)
                    if corners_history:
                        average_corners = np.mean(corners_history, axis=0)
                        print(f"Calculated average corners from {len(corners_history)} valid frames")
                        states, visualized, _ = process_frame(
                            adjusted, grid_params,
                            debug_visualizer=debug_visualizer,
                            override_corners=average_corners,
                            game_state=game_state,
                            game_controller=game_controller,
                            frame_number=debug_frame,
                            detection_config=detection_config
                        )
                    else:
                        print("No valid corners found in first frames")
                        _, visualized, _ = process_frame(
                            adjusted, grid_params,
                            debug_visualizer=debug_visualizer,
                            game_state=game_state,
                            game_controller=game_controller,
                            frame_number=debug_frame,
                            detection_config=detection_config
                        )
                else:
                    # For frames before frames, process normally
                    _, _, prev_gray, prev_points = preprocess_frame(prev_frame, None, None, debug_visualizer) if prev_frame is not None else (None, None, None, None)
                    adjusted, transform, gray, points = preprocess_frame(frame, prev_gray, prev_points, debug_visualizer)
                    _, visualized, _ = process_frame(
                        adjusted, grid_params,
                        debug_visualizer=debug_visualizer,
                        transform=transform,
                        game_state=game_state,
                        game_controller=game_controller,
                        frame_number=debug_frame,
                        detection_config=detection_config
                    )
            cv2.destroyAllWindows()
        else:
            # Process all frames
            frame_count = video_handler.frame_count
            corners_history = []  # Store corners from first frames
            average_corners = None
            prev_frame = None
            prev_gray = None
            prev_points = None
            
            with tqdm(total=frame_count, desc="Processing video") as pbar:
                frame_number = 0
                while frame_number < frame_count:
                    try:
                        frame = video_handler.extract_frame(frame_number)
                        if frame is None:
                            break
                            
                        # Preprocess frame
                        adjusted, transform, gray, points = preprocess_frame(frame, prev_gray, prev_points, debug_visualizer)
                        
                        # Process frame and get corners
                        if frame_number < FRAMES_TO_PROCESS:
                            _, visualized, corners = process_frame(
                                adjusted, grid_params,
                                debug_visualizer=debug_visualizer,
                                transform=transform,
                                game_state=game_state,
                                game_controller=game_controller,
                                frame_number=frame_number,
                                detection_config=detection_config
                            )
                            # Store corners if they were detected
                            if corners is not None:
                                corners_history.append(corners)
                            
                            # Calculate average corners when we have frames
                            if frame_number == FRAMES_TO_PROCESS - 1 and corners_history:
                                # Calculate average corners
                                average_corners = np.mean(corners_history, axis=0)
                                print(f"Calculated average corners from {len(corners_history)} valid frames")
                        else:
                            # For frames after frames, use the average corners
                            _, visualized, _ = process_frame(
                                adjusted, grid_params,
                                debug_visualizer=debug_visualizer,
                                override_corners=average_corners,
                                transform=transform,
                                game_state=game_state,
                                game_controller=game_controller,
                                frame_number=frame_number,
                                detection_config=detection_config
                            )
                        
                        if video_writer is not None and visualized is not None:
                            video_writer.write(visualized)
                            
                        if display and visualized is not None:
                            cv2.imshow('Game Analysis', visualized)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        prev_frame = frame
                        prev_gray = gray
                        prev_points = points
                                
                    except Exception as e:
                        import traceback
                        print(f"Error processing frame {frame_number}: {str(e)}")
                        print("Traceback:")
                        traceback.print_exc()
                        
                    frame_number += 1
                    pbar.update(1)
                    
            if display:
                cv2.destroyAllWindows()
    
    finally:
        if video_writer is not None:
            video_writer.release()
        video_handler.cap.release()

def preprocess_frame(frame: np.ndarray,
                    prev_gray: Optional[np.ndarray] = None,
                    prev_points: Optional[np.ndarray] = None,
                    debug_visualizer: Optional[DebugVisualizer] = None) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Preprocess frame with stabilization, noise reduction, and lighting adjustment."""
    # Resize frame to 1/10th of original size
    height, width = frame.shape[:2]
    new_width = width // 10
    new_height = height // 10
    resized = frame #cv2.resize(frame, (new_width, new_height))
    
    debug_visualizer.show("1. Original Frame", frame) if debug_visualizer else None
    # debug_visualizer.show("2. Resized Frame", resized) if debug_visualizer else None
    
    # Stabilize
    transform, gray, points = estimate_transform(resized, prev_gray, prev_points, debug_visualizer)
    if transform is not None:
        rows, cols = resized.shape[:2]
        stabilized = cv2.warpAffine(resized, transform, (cols, rows))
        debug_visualizer.show("1.4 Stabilized Frame", stabilized) if debug_visualizer else None
    else:
        stabilized = resized
    
    # Denoise
    denoised = apply_gaussian(stabilized, debug_visualizer)
    debug_visualizer.show("1.5 Denoised Frame", denoised) if debug_visualizer else None
    
    # Adjust lighting
    adjusted = apply_clahe(denoised, debug_visualizer)
    debug_visualizer.show("1.9 CLAHE Adjusted Frame", adjusted) if debug_visualizer else None
    
    return adjusted, transform, gray, points

def process_frame(frame: np.ndarray,
                 grid_params: Dict,
                 debug_visualizer: Optional[DebugVisualizer] = None,
                 transform: Optional[np.ndarray] = None,
                 override_corners: Optional[np.ndarray] = None,
                 counter_debug: bool = False,
                 game_state: Optional[GameState] = None,
                 game_controller: Optional[GameController] = None,
                 frame_number: int = 0,
                 detection_config: Optional[Dict] = None) -> Tuple[List[CounterState], np.ndarray, Optional[np.ndarray]]:
    """Process a single frame."""
    
    # Use empty dict if no config provided
    detection_config = detection_config or {}
    
    # Update frame number in game controller
    if game_controller:
        game_controller.frame_number = frame_number
    
    if override_corners is not None:
        # Transform override_corners to match the adjusted frame coordinates
        if transform is not None:
            # Convert corners to homogeneous coordinates
            corners_h = np.hstack([override_corners, np.ones((override_corners.shape[0], 1))])
            # Apply the transform to get corners in adjusted frame coordinates
            corners = np.dot(corners_h, transform.T)
        else:
            corners = override_corners
    else:
        # Step 1: Create grid mask using color thresholding
        grid_mask = create_grid_mask(frame, grid_params['grid_color_range']['lower'], grid_params['grid_color_range']['upper'])
        if debug_visualizer:
            debug_visualizer.show("Grid Mask", grid_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((10,10), np.uint8)
        grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, kernel)
        
        if debug_visualizer:
            debug_visualizer.show("Grid Mask - Cleaned", grid_mask)
        
        # Step 2: Find grid corners using contour detection
        contour = find_largest_contour(grid_mask)
        if contour is None:
            print("Failed to find grid contour")
            return None, None, None
            
        if debug_visualizer:
            contour_viz = frame.copy()
            cv2.drawContours(contour_viz, [contour], -1, (0, 255, 0), 2)
            debug_visualizer.show("Grid Contour", contour_viz)
            
        # # Create a mask from the contour for morphological processing
        # x, y, w, h = cv2.boundingRect(contour)
        # mask = np.zeros((h + 10, w + 10), dtype=np.uint8)
        # shifted_contour = contour - np.array([x - 5, y - 5])
        # cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
        
        # # Apply morphological closing to smooth the contour
        # kernel = np.ones((10, 10), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=2)
        
        # if debug_visualizer:
        #     debug_visualizer.show("Processed Contour Grid Mask", mask)
        
        # # Find contours in the processed mask
        # processed_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if not processed_contours:
        #     print("Failed to find processed contours")
        #     return None, None, None
            
        # processed_contour = max(processed_contours, key=cv2.contourArea)
        # processed_contour = processed_contour + np.array([x - 5, y - 5])
        
        # if debug_visualizer:
        #     processed_viz = adjusted.copy()
        #     cv2.drawContours(processed_viz, [processed_contour], -1, (0, 255, 0), 2)
        #     debug_visualizer.show("Processed Contour", processed_viz)
            
        # Find corner points
        corners = find_corner_points(contour, grid_params['epsilon_factor'])
        if corners is None or len(corners) != 4:
            print("Failed to find 4 corners")
            return None, None, None
        
        # Sort corners in clockwise order
        corners = sort_corners(corners.reshape(-1, 2))
    
    if debug_visualizer:
        corners_viz = frame.copy()
        # Draw lines between corners
        for i in range(4):
            cv2.line(corners_viz,
                    tuple(corners[i].astype(int)),
                    tuple(corners[(i+1)%4].astype(int)),
                    (0, 255, 0), 2)
        
        # Draw corner points with labels
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (corner, label) in enumerate(zip(corners, labels)):
            point = tuple(corner.astype(int))
            cv2.circle(corners_viz, point, 5, (0, 0, 255), -1)
            cv2.putText(corners_viz, label, 
                       (point[0] + 10, point[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        debug_visualizer.show("Corner Points", corners_viz)
    
    # Get cell centers
    centers = get_cell_centers(corners, grid_params['rows'], grid_params['cols'])
    
    # Create color masks for the entire frame
    red_mask, yellow_mask = create_color_masks(
        frame,
        grid_params['red_ranges'][0]['lower'],
        grid_params['red_ranges'][0]['upper'],
        grid_params['red_ranges'][1]['lower'],
        grid_params['red_ranges'][1]['upper'],
        grid_params['yellow_range']['lower'],
        grid_params['yellow_range']['upper']
    )
    
    # if debug_visualizer:
    #     debug_visualizer.show("Red Mask", red_mask)
    #     debug_visualizer.show("Yellow Mask", yellow_mask)
    
    # Step 5: Detect counter colors for each cell
    states = []
    for center in centers:
        state = detect_counter_color(
            frame,
            center,
            grid_params['circle_detection']['min_radius'],  # fallback radius if needed
            grid_params['red_ranges'][0]['lower'],
            grid_params['red_ranges'][0]['upper'],
            grid_params['red_ranges'][1]['lower'],
            grid_params['red_ranges'][1]['upper'],
            grid_params['yellow_range']['lower'],
            grid_params['yellow_range']['upper'],
            grid_params['cols'],
            grid_params['rows'],
            grid_params['color_threshold'],
            corners,  # pass corners for ROI calculation
            pre_calculated_masks=(red_mask, yellow_mask),  # pass pre-calculated masks
            debug_visualizer=debug_visualizer if counter_debug else None
        )
        states.append(state)
    
    # Process dice
    dice_value = detect_dice(frame, detection_config.get('dice', {}), corners, debug_visualizer)
    permanent_dice = game_controller.process_dice(dice_value) if game_controller else 0

    # Detect counters on bench
    (yellow_count, yellow_centers), (red_count, red_centers) = detect_bench_counters(
        frame, corners, yellow_mask, red_mask, debug_visualizer)
    if debug_visualizer:
        print(f"Bench counters - Yellow: {yellow_count}, Red: {red_count}")
        
    # Process bench positions
    if game_controller:
        yellow_centers, red_centers = game_controller.process_bench(yellow_centers, red_centers)
    
    # Create visualization
    visualized_frame = visualize_grid(
        frame, corners, states, grid_params, 
        permanent_dice=permanent_dice,
        game_state=game_state,
        game_controller=game_controller,
        bench_centers={'yellow': yellow_centers, 'red': red_centers}
    )
    debug_visualizer.show("Final Result", visualized_frame) if debug_visualizer else None
    
    return states, visualized_frame, corners

def visualize_grid(frame: np.ndarray,
                  corners: np.ndarray,
                  states: list,
                  grid_params: dict,
                  permanent_dice: int = 0,
                  game_state: GameState = None,
                  game_controller: GameController = None,
                  bench_centers: dict = None) -> np.ndarray:
    """Visualize the game grid with counters and dice value."""
    visualized_frame = frame.copy()
    
    if corners is not None:
        # Draw grid lines
        for i in range(4):
            cv2.line(visualized_frame,
                    tuple(corners[i].astype(int)),
                    tuple(corners[(i+1)%4].astype(int)),
                    (0, 255, 0), 2)
        
        # Draw cell states
        centers = get_cell_centers(corners, grid_params['rows'], grid_params['cols'])
        left, top = corners[0]
        right, bottom = corners[2]
        cell_width = (right - left) / grid_params['cols']
        cell_height = (bottom - top) / grid_params['rows']
        radius = int(min(cell_width, cell_height) * 0.4)
        
        # Validate radius
        if radius <= 0:
            print(f"Warning: Invalid radius {radius} calculated from cell dimensions {cell_width}x{cell_height}")
            return visualized_frame
            
        # First draw all permanent counters
        if game_state:
            for row in range(grid_params['rows']):
                for col in range(grid_params['cols']):
                    if game_state.is_permanent(row, col):
                        idx = row * grid_params['cols'] + col
                        center = centers[idx]
                        state_value = game_state.grid[row][col]
                        
                        # Set color based on the stored state
                        if state_value == CounterState.RED.value:
                            color = (0, 0, 255)  # BGR format
                        else:  # YELLOW
                            color = (0, 255, 255)
                            
                        cv2.circle(visualized_frame, center, radius, color, -1)
                        
                        # Draw border - green for permanent, yellow for winning
                        if game_controller and game_controller.winning_cells and (row, col) in game_controller.winning_cells:
                            cv2.circle(visualized_frame, center, radius, (255, 0, 0), 4)  # Blue thick border
                        else:
                            cv2.circle(visualized_frame, center, radius, (0, 255, 0), 3)  # Green thick border
                            
            # Draw line through winning cells if they exist
            if game_controller and game_controller.winning_cells and len(game_controller.winning_cells) >= 2:
                # Get the centers of the first and last winning cells
                start_row, start_col = game_controller.winning_cells[0]
                end_row, end_col = game_controller.winning_cells[-1]
                # Convert to grid coordinates
                start_idx = start_row * grid_params['cols'] + start_col
                end_idx = end_row * grid_params['cols'] + end_col
                
                # Ensure indices are within bounds
                if (0 <= start_idx < len(centers) and 
                    0 <= end_idx < len(centers)):
                    start_center = centers[start_idx]
                    end_center = centers[end_idx]
                
                # Draw thick line through winning cells
                cv2.line(visualized_frame,
                    tuple(map(int, start_center)),
                    tuple(map(int, end_center)),
                    (255, 0, 0), 5)  # Blue thick line
        
        # Then process current frame detections for non-permanent counters
        if game_state:
            for idx, (center, state) in enumerate(zip(centers, states)):
                row = idx // grid_params['cols']
                col = idx % grid_params['cols']
                
                # Skip if this position is already permanent
                if game_state.is_permanent(row, col):
                    continue
                
                if state == CounterState.RED:
                    color = (0, 0, 255)  # BGR format
                    success = game_state.track_counter(row, col, CounterState.RED.value)
                    if success and game_controller:
                        game_controller.process_move(row, col, CellState.RED)
                elif state == CounterState.YELLOW:
                    color = (0, 255, 255)
                    success = game_state.track_counter(row, col, CounterState.YELLOW.value)
                    if success and game_controller:
                        game_controller.process_move(row, col, CellState.YELLOW)
                else:
                    # Draw empty cell marker
                    cv2.circle(visualized_frame, center, radius,
                            (128, 128, 128), 2)
                    continue
                    
                # Draw non-permanent counter
                cv2.circle(visualized_frame, center, radius, color, -1)
                cv2.circle(visualized_frame, center, radius, (255, 255, 255), 2)  # White normal border
    
    # Draw bench counter numbers
    if bench_centers:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        def draw_number(img, text, pos):
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Calculate background rectangle coordinates
            x, y = pos
            padding = 5
            bg_x1 = x - padding
            bg_y1 = y - text_height - padding
            bg_x2 = x + text_width + padding
            bg_y2 = y + padding
            
            # Draw black background rectangle
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            # Draw white text
            cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Draw yellow counter numbers
        for i, center in enumerate(bench_centers.get('yellow', []), 1):
            draw_number(visualized_frame, str(i), (center[0]-10, center[1]+10))
        
        # Draw red counter numbers
        for i, center in enumerate(bench_centers.get('red', []), 1):
            draw_number(visualized_frame, str(i), (center[0]-10, center[1]+10))
    
    # Draw dice value
    height, width = frame.shape[:2]
    cv2.putText(visualized_frame, f"Dice: {permanent_dice}",
                (width*3//4, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    return visualized_frame

if __name__ == "__main__":
    main()
